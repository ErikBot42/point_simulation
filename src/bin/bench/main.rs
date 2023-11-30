use point_simulation::CpuSimulationState;
use std::hint::black_box;
use std::time::Instant;
fn main() {
    let mut state = black_box(CpuSimulationState::from_seed(23483489));
    //let iters = 10000;
    //let iters_pre = 1000;
    let iters = 50000;
    let iters_pre = 100;
    for _ in 0..iters_pre{
        state.update();
    }
    let now = Instant::now();
    for _ in 0..iters {
        state.update();
    }
    let elapsed = now.elapsed().as_secs_f64();
    println!("{} ups, {} s", iters as f64 / elapsed, elapsed);
    black_box(state);

}
