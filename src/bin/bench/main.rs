use point_simulation::SimulationState;
use std::hint::black_box;
use std::time::Instant;
fn main() {
    let mut state = black_box(SimulationState::from_seed(23483489));
    let iters = 1000;
    for _ in 0..100 {
        state.update();
    }
    let now = Instant::now();
    for _ in 0..iters {
        state.update();
    }
    println!("{} ups", iters as f64 / now.elapsed().as_secs_f64());
    black_box(state);

}
