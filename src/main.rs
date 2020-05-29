use std::process::Stdio;
use tokio::process::Command;
use tokio::io::AsyncWriteExt;
use tokio::time::timeout;
use futures::future::{select_all};
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

    let max_time = std::time::Duration::from_millis(2000);

    // steps 1..cutoff, normal incremental solver
    let command = format!(
        "go_bmc\ncheck_ltlspec_bmc_inc -k {}\nshow_traces -v\nquit\n",
        cutoff
    );
    let fut = Box::pin(call_nuxmv(filename.clone(), command, cutoff));
    tasks.push(timeout(max_time, WrappedWorkTask::from(cutoff, fut)));

    // solve one instance at a time concurrently for bounds above the cutoff
    for i in (cutoff+1)..max_steps {
        let command = format!(
            "go_bmc\ncheck_ltlspec_bmc_onepb -k {}\nshow_traces -v\nquit\n",
            i
        );
        let fut = Box::pin(call_nuxmv(filename.clone(), command, i));
        tasks.push(timeout(max_time, WrappedWorkTask::from(i, fut)));
    }

    let now = std::time::Instant::now();
    let mut one = select_all(tasks);
    loop {
        let (x, _, mut remaining) = one.await;

        let (c,r,e) = match x {
            Result::Ok(Result::Ok((c,r,e))) => (c,r,e),
            _ => {
                if remaining.is_empty() {
                    return None;
                }
                one = select_all(remaining);
                continue;
            }
        };

        if !r.contains("Trace Type: Counterexample") {
            println!("no plan found for bound {} after {}ms", c, now.elapsed().as_millis());
            remaining.retain(|f| f.get_ref().max_steps > c); // stop looking for shorter plans
            if remaining.is_empty() {
                return None;
            }
            one = select_all(remaining);
        } else {
            println!("first guess found after after {}ms (with plan length: {})",
                     now.elapsed().as_millis(), c);

            let dur = now.elapsed().mul_f32(lookout);
            println!("setting timeout to {}", dur.as_millis());
            let remaining: Vec<_> = remaining
                .into_iter()
                .filter(|f| f.get_ref().max_steps < c) // stop looking for longer plans
                .map(|r| timeout(dur, r)).collect();

            if remaining.is_empty() {
                // no more tasks to run!
                return Some((c,r,e))
            }

            let mut solutions = vec![(c,r,e)];
            let mut one = select_all(remaining);
            loop {
                let (x, _, mut remaining) = one.await;
                match x {
                    // outer timeout, io error, inner (global) timeout
                    Result::Ok(Result::Ok(Result::Ok((c,r,e)))) => {
                        if r.contains("Trace Type: Counterexample") {
                            // save solution
                            solutions.push((c,r,e));
                            // stop looking for longer plans
                            remaining.retain(|f| f.get_ref().get_ref().max_steps < c);
                        } else {
                            // stop looking for shorter plans
                            remaining.retain(|f| f.get_ref().get_ref().max_steps > c);
                        }
                    }
                    _ => {}
                }
                if remaining.is_empty() {
                    break;
                }
                one = select_all(remaining);
            }
            return solutions.into_iter().min_by(|x,y| x.0.cmp(&y.0));
        }
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
    //let filename: &str = "small_problem.bmc";

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
    println!("heuristic search done after {}ms", heuristic_time);

    result.iter().for_each(|(c,_r,_)| {
        println!("plan length: {}", c+1);
    });

    println!("------");
    let speedup = classic_time as f32 / heuristic_time as f32 - 1f32;
    println!("speedup {:.2}x", speedup);

    Ok(())
}
