use point_simulation::CpuSimulationState;
use point_simulation::NUM_POINTS;
use std::arch::x86_64::_rdtsc;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    // assert_ne!(libc::nice(-50), -1, "could not set nice level");
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(2, &mut set);
        libc::CPU_SET(3, &mut set);
        assert_eq!(
            libc::sched_setaffinity(
                libc::getpid(),
                std::mem::size_of::<libc::cpu_set_t>(),
                (&set) as *const _
            ),
            0
        );
    }
    let mut state = black_box(CpuSimulationState::from_seed(23483489));
    //let iters = 10000;
    //let iters_pre = 1000;
    let iters = 5000;
    let iters_pre = 100;

    let frequency: u64 = 3100 * 1000 * 1000;

    for _ in 0..iters_pre {
        state.update();
    }

    let cycles_pre = unsafe { _rdtsc() };
    let now = Instant::now();
    for _ in 0..iters {
        state.update();
    }
    let cycles_post = unsafe { _rdtsc() };
    let elapsed = now.elapsed().as_secs_f64();

    let update_points = NUM_POINTS as f64 * iters as f64;
    let cycles_per_update_rdtsc =
        ((cycles_post - cycles_pre) as f64) / update_points;
    let cycles_time = elapsed * frequency as f64;
    let cycles_per_update_time = cycles_time / update_points;

    eprintln!(
        "{} ups\n{} s\n{} cycles per update (rdtsc)\n{} cycles per update (time)",
        iters as f64 / elapsed,
        elapsed,
        cycles_per_update_rdtsc,
        cycles_per_update_time,
    );
    black_box(state);
}
