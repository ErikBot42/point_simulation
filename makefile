
hotspot: perf.data
	hotspot perf.data

run_bench: target/release/bench
	./target/release/bench
	./target/release/bench

perf.data: target/release/bench makefile
	#perf record --call-graph dwarf,8192 -e branch-misses,cycles,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-stores,L1-icache-load-misses,LLC-load-misses,LLC-loads,LLC-store-misses,LLC-stores target/release/bench
	perf record --call-graph dwarf,8192 -e cycles target/release/bench

target/release/bench: makefile src/
	RUSTFLAGS="-C target-cpu=native" cargo build --release --bin bench
