use std::process::Stdio;
use tokio::process::Command;
use tokio::io::AsyncWriteExt;
use tokio::time::timeout;
use futures::future::{select_all, join_all};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

type NuxmvOutput = (u32, String, String);

pub struct WrappedWorkTask {
    max_steps: u32,
    inner: Pin<Box<dyn Future<Output = std::io::Result<NuxmvOutput>>>>,
}

impl Future for WrappedWorkTask {
    type Output = std::io::Result<NuxmvOutput>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.inner.as_mut().poll(cx)
    }
}

impl WrappedWorkTask {
    fn from(max_steps: u32, inner:
            Pin<Box<dyn Future<Output = std::io::Result<NuxmvOutput>>>>) -> Self {
        Self {
            max_steps,
            inner
        }
    }
}

async fn call_nuxmv(filename: String, command: String, max_len: u32) ->
    std::io::Result<NuxmvOutput> {
    let mut process = Command::new("nuXmv")
        .arg("-int")
        .arg(filename)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()?;

    let mut stdin = process.stdin.take().unwrap();
    stdin.write(command.as_bytes()).await?;

    let result = process.wait_with_output().await?;

    assert!(result.status.success());

    let raw = String::from_utf8(result.stdout).expect("Check character encoding");
    let raw_error = String::from_utf8(result.stderr).expect("Check character encoding");

    Ok((max_len, raw, raw_error))
}

/// Look for plans concurrently.
/// We start one process that searches incrementally from step 1 up to `cutoff`.
/// At the sime time, we start multiple processes for plan lengths of cutoff+1 to max_steps.
/// Whenever the first solution has been found, kill the processes that we know are looking
/// for plans with either to few or too many step. Let the remaining processes run for the
/// time it has taken to find the first guess times the `lookout`-factor. When this time is up,
/// return the solution with the shortest plan length.
async fn search_heuristic(filename: String,
                          cutoff: u32,
                          max_steps: u32,
                          lookout: f32) -> Option<NuxmvOutput> {
    let mut tasks = Vec::new();

    // steps 1..cutoff, normal incremental solver
    let command = format!(
        //"go_bmc\ncheck_ltlspec_bmc_inc -k {}\nshow_traces -v\nquit\n",
        "go_bmc\ncheck_ltlspec_bmc_inc -k {}\nquit\n",
        cutoff
    );
    let fut = Box::pin(call_nuxmv(filename.clone(), command, cutoff));
    tasks.push(WrappedWorkTask::from(cutoff, fut));

    // solve one instance at a time concurrently for bounds above the cutoff
    for i in (cutoff+1)..max_steps {
        let command = format!(
            //"go_bmc\ncheck_ltlspec_bmc_onepb -k {}\nshow_traces -v\nquit\n",
            "go_bmc\ncheck_ltlspec_bmc_onepb -k {}\nquit\n",
            i
        );
        let fut = Box::pin(call_nuxmv(filename.clone(), command, i));
        tasks.push(WrappedWorkTask::from(i, fut));
    }

    let now = std::time::Instant::now();
    let mut one = select_all(tasks);
    loop {
        let (x, _, remaining) = one.await;

        let (c,r,_e) = x.unwrap();
        if !r.contains("no counterexample found with bound") {
            println!("first guess found after after {}ms (with plan length: {})",
                     now.elapsed().as_millis(), c);

            let dur = now.elapsed().mul_f32(lookout);
            println!("setting timeout to {}", dur.as_millis());
            let new_futs: Vec<_> = remaining
                .into_iter()
                .filter(|f| f.max_steps < c) // stop looking for longer plans
                // .map(|r| {println!("waiting for {} still", r.max_steps); r})
                .map(|r| timeout(dur, r)).collect();
            let results = join_all(new_futs).await;

            let x = results
                .into_iter()
                .filter_map(Result::ok) // throw away timeouts
                .filter_map(Result::ok) // throw away inner IO errors
                .filter(|(_,r,_)|
                        !r.contains("no counterexample found with bound"))
                .min_by(|x,y| x.0.cmp(&y.0)); // keep only solution with lowest bound

            return x;
        }

        println!("no plan found for bound {} after {}ms", c, now.elapsed().as_millis());

        let remaining: Vec<_> = remaining
            .into_iter()
            .filter(|f| f.max_steps > c) // stop looking for shorter plans
            // .map(|r| {println!("waiting for {} still", r.max_steps); r})
            .collect();

        if remaining.is_empty() {
            // no soultion found
            return None;
        }
        one = select_all(remaining);
    }
}

async fn without_heuristic(filename: String) -> std::io::Result<String> {
    let max_steps = 40;
    let command = format!(
        //"go_bmc\ncheck_ltlspec_bmc_inc -k {}\nshow_traces -v\nquit\n",
        "go_bmc\ncheck_ltlspec_bmc_inc -k {}\nquit\n",
        max_steps
    );
    let (_c,r,_e) = call_nuxmv(filename, command, max_steps).await?;
    Ok(r)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut runtime = tokio::runtime::Runtime::new().unwrap();

    let filename: &str = "big_op_model.bmc";

    // first run plain old incremental to find answer
    println!("start classic incremental");
    let now = std::time::Instant::now();
    let r = runtime.block_on(without_heuristic(filename.to_owned()))?;
    let plan_length = r.lines().rev().find(|l|l.contains("State:")).unwrap();
    let plan_length = &plan_length[14..plan_length.len()-3];
    let classic_time = now.elapsed().as_millis();
    println!("classic search done after {}ms", classic_time);
    println!("plan length {}", plan_length);

    println!("------");

    println!("start concurrent heuristic");
    let now = std::time::Instant::now();
    let result = runtime.block_on(search_heuristic(filename.to_owned(), 15, 40, 1.5));

    let heuristic_time = now.elapsed().as_millis();
    result.iter().for_each(|(c,_r,_)| {
        println!("heuristic search done after {}ms", heuristic_time);
        println!("plan length: {}", c+1);
    });

    println!("------");
    println!("speedup {:.2}x", classic_time as f32 / heuristic_time as f32);

    Ok(())
}
