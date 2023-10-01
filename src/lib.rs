#![feature(portable_simd)]
use std::array::from_fn;
use std::marker::PhantomData;
use std::mem::{replace, size_of, swap};
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Index;
use std::ops::IndexMut;
use std::time::Duration;
use wgpu::util::DeviceExt;
use wgpu::SurfaceError;
use winit::window::Window;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use prng::Prng;

mod prng;
mod vertex;

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Instant, SystemTime};

const CELLS_X: u16 = 64;
const CELLS_Y: u16 = 64;
const CELLS_MAX_DIM: u16 = if CELLS_X > CELLS_Y { CELLS_X } else { CELLS_Y };
const CELLS: u16 = CELLS_X * CELLS_Y;
const POINT_MAX_RADIUS: f32 = 1.0 / (CELLS_MAX_DIM as f32);

const DT: f32 = 0.004; // TODO: define max speed/max dt from cell size.
const NUM_POINTS_U: usize = NUM_POINTS as usize;
const CELLS_PLUS_1_U: usize = CELLS as usize + 1;
const KINDS2: usize = NUM_KINDS as usize * NUM_KINDS as usize;
const NUM_POINTS: u32 = 8192; //16384;
const NUM_KINDS: usize = 8;

fn compute_cell(x: f32, y: f32) -> Cid {
    // x in 0_f32..1_f32
    // y in 0_f32..1_f32
    let cell_x = ((x * CELLS_X as f32) as u16).min(CELLS_X);
    let cell_y = ((y * CELLS_Y as f32) as u16).min(CELLS_Y);
    let cell = cell_x + cell_y * CELLS_X;
    debug_assert!(
        cell < CELLS,
        "{cell} >= CEllS, cell_x: {cell_x}, cell_y: {cell_y}, x: {x}, y: {y}"
    );
    Cid(cell)
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct EulerPoint {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
}
unsafe impl bytemuck::Zeroable for EulerPoint {}
unsafe impl bytemuck::Pod for EulerPoint {}

// v0 and v1 will be indexed randomly with gather instruction.
// we want to read x and y simultaneously
//
// register(indexes to v0)
// -> register(xs), register(ys)
// -> reduction
// -> single scalar write to v1
//
// we can perform two 64 bit gathers and interleave the results.
// i0, i1, i2, i3
// i4, i5, i6, i7
//
// ->
// (x0 y0) (x1 y1) (x2 y2) (x3 y3)
// (x4 y4) (x5 y5) (x6 y6) (x7 y7)
// ->
// x0 x1 x2 x3 x4 x5 x6 x7
// y0 y1 y2 y3 y4 y5 y6 y7
//
//
/*
DEFINE INTERLEAVE_DWORDS(src1[127:0], src2[127:0]) {
    dst[31:0] := src1[31:0]
    dst[63:32] := src2[31:0]
    dst[95:64] := src1[63:32]
    dst[127:96] := src2[63:32]
    RETURN dst[127:0]
}
dst[127:0] := INTERLEAVE_DWORDS(a[127:0], b[127:0])
dst[255:128] := INTERLEAVE_DWORDS(a[255:128], b[255:128])
dst[MAX:256] := 0

src1 = x0 y0 x1 y1
src2 = x4 y5 x6 y6


actual order of values is mostly irrelevant

all indices are 32-bit even if elements are 64-bit

-> no need to do interleave tricks with the indexes, just shift upper half (probably specific instruction for this though)


avaliable shuffling instructions:

_mm256_blend_ps/vblendps ymm, ymm, ymm, imm8: (0.33/0.33, 1*p015)
select based on imm8 between two registers. essentially just a mask operation
this but variable exists but is high latency

_mm256_permute_ps/vpermilps ymm, ymm, imm8: (1.0/1.0, 1*p5)
within each 128-bit group, perform arbitrary permute.

_mm256_permutevar8x32_ps/vpermps ymm, ymm, ymm: (1.0/1.0, 1*p5)
generic 32-bit shuffle based on dynamic 32-bit indexes.

strategies:
circular shift r1 by 1 and use blend ops to produce reg with xs and ys

x0 y0 x1 y1 x2 y2 x3 y3
x4 y4 x5 y5 x6 y6 x7 y7

x0    x1    x2    x3
   y0    y1    y2    y3

x4    x5    x6    x7
   y4    y5    y6    y7

x0    x1    x2    x3
y0    y1    y2    y3

   x4    x5    x6    x7
   y4    y5    y6    y7

x0 x4 x1 x5 x2 x6 x3 x7
y0 y4 y1 y5 y2 y6 y3 y7

x0 y0 x1 y1 x2 y2 x3 y3
x4 y4 x5 y5 x6 y6 x7 y7

x0 x1 x2 x3 x4 x5 x6 x7
y0 y1 y2 y3 y4 y5 y6 y7

x0 y0 x1 y1 x2 y2 x3 y3
x4 y4 x5 y5 x6 y6 x7 y7

x0 x2 x4 x6 y0 y2 y4 y6
x1 x3 x5 x7 y1 y3 y5 y7
-> deinterleave ->
x0 x1 x2 x3 x4 x5 x6 x7
y0 y1 y2 y3 y4 y5 y6 y7

instead of doing a gather:
just store point *type* and xy location. ids no longer matter. access pattern will be ideal since writes after accumulation will be localized.

It may be possible to embed type in the point itself (trade mem with compute).
An open question now is if AoS or SoA should be used. => num points and point types can be dynamic.

IEEE-754 32 bit floats:
1 bit sign
8 bit exponent
23 bit fraction

only one time bounds check calculations care about the actual play-field.

sign, exponent:
in 0-1:
0, [0, 127] -> 2 bits unused

in 1-2:
0, [127, 128] -> 8 bits of information unused, 7 relatively packed.

in 2-4:
0, [128, 129] -> 6 packed unused bits -> 32 point types, 12 total -> 4096 types.


16-bit fixed point:
0.0 (0) <-> 1.0 (65535)
This produces 512 possible points within a cell in a 128x128 grid.

expand/compress to f32?
fast sqrt needed for distance calc.



spatial hash -> linear:

struct IterationInfo {
    neighborhood: &[(T;T)],
    this: &[(T;T)]
}


hash table strategies:

I want to scale to 65536 points, with a 128x128 grid.
assume 8 bytes per point/object (2 floats).

naive preallocate:
2d linear array where each cell has a 65536 element vector => 1 GiB + a bunch of associativity conflicts.

paper 1: https://www.researchgate.net/publication/277870601
A Hash Table Construction Algorithm for Spatial Hashing Based on Linear
Memory

calc cell for each point.
struct Cell {
    points: usize,
    start: usize,
    end: usize,
}

all point data is then stored in an array.


improvement to paper 1:
use a Csr

struct SpaceHash {
    counts: Vec<usize>, // # = |cells|
    indexes: Vec<usize> // # = |cells|
    data: Vec<T>,
}
// given original constraints
struct SpaceHash {
    counts: Vec<u16>, // # = |cells|
    indexes: Vec<u16> // # = |cells|
    data: Vec<(f32, f32)>,
}
MEM(n) = cells * 4 + points * 8
scratch(n) = points * 10

this can be improved by replacing counts with indexes in a single pass
struct SpaceHash {
    counts_or_indexes: Vec<u16>, // # = |cells|
    data: Vec<(f32, f32)>,
}
MEM(n) = cells * 2 + points * 8
scratch(n) = points * 10


still need a 2 step process though.

benefit of a Csr style is that SIMD ops can be more linear

dumping all of this into a linearly processable array is trivial

(aaaaaBBBBcccccc
ddd(EEE)fffff
ggggHHHiiiii)
kkkkLLLLmmmm

the linearly processable array will use 3x the memory though


problem: linear processable array wastes memory
solution: write backwards if enough space

aaaAAAaaa
bbbBBBbbb // clear and start writing eeee backwards
cccCCCccc
dddDDDddd


TODO: persistent counts and use same count for both pairs of the iteration? - do later maybe


the buffer would need to be |points| * 4 length, but that should be fine.


Store separate buffers for x and y, deinterleave requires a ton of needless instructions.
It may be beneficial to offset the allocations of the x and y buffers so there are no cache conflicts between them when iterating linearly.

TODO: need previous position somehow

 * */
//
//
// TODO: maybe merge position and kind for ideal cache properties.
// TODO: make this allocation clever and external.
/*
// spacehash_indexes -> tmpstate -> spacehash_counts -> spacehash_indexes
struct SpaceHash {
    counts_or_indexes: Vec<u16>, // |cells|
    data_x: Vec<f32>,            // |points|
    data_y: Vec<f32>,            // |points|
    data_prev_x: Vec<f32>,       // |points|
    data_prev_y: Vec<f32>,       // |points|
    kind: Vec<u8>,               // |points|
}
struct TmpState {
    cell: Vec<u16>,              // |points|
    data_x: Vec<f32>,            // |points|
    data_y: Vec<f32>,            // |points|
    data_prev_x: Vec<f32>,       // |points|
    data_prev_y: Vec<f32>,       // |points|
    kind: Vec<u8>,               // |points|
}
*/

// if we keep indexes into the previous iteration instead:
struct SpaceHash {
    counts_or_indexes: Vec<u16>, // |cells|
    data_x: Vec<f32>,            // |points|
    data_y: Vec<f32>,            // |points|
    prev_data_x: Vec<f32>,       // |points|
    prev_data_y: Vec<f32>,       // |points|
    kind: Vec<u8>,               // |points|
}
struct TmpState {
    cell: Vec<u16>, // |points|
}

// the randomly generated parts of the simulation
struct SimulationInputEuler {
    x: Pbox<f32>,
    y: Pbox<f32>,
    vx: Pbox<f32>,
    vy: Pbox<f32>,
    k: Pbox<u8>,
    relation_table: Box<[f32; KINDS2]>,
}
struct SimulationInputVerlet {
    x0: Pbox<f32>,
    y0: Pbox<f32>,
    x1: Pbox<f32>,
    y1: Pbox<f32>,
    k: Pbox<u8>,
    relation_table: Box<[f32; KINDS2]>,
}
impl SimulationInputEuler {
    fn new() -> SimulationInputEuler {
        let mut rng = Prng(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64,
        );
        let xy_range = 0.0..1.0;
        let v_range = -1.0..1.0;
        fn iter_alloc<T: std::fmt::Debug, const SIZE: usize, F: FnMut() -> T>(
            mut f: F,
        ) -> Box<[T; SIZE]> {
            (0..SIZE)
                .map(|_| f())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        }

        SimulationInputEuler {
            x: iter_alloc(|| rng.f32(xy_range.clone())).into(),
            y: iter_alloc(|| rng.f32(xy_range.clone())).into(),
            vx: iter_alloc(|| rng.f32(v_range.clone())).into(),
            vy: iter_alloc(|| rng.f32(v_range.clone())).into(),
            k: iter_alloc(|| rng.u8(NUM_KINDS as u8)).into(),
            relation_table: iter_alloc(|| rng.f32(-1.0..1.0)),
        }
    }
    fn verlet(self) -> SimulationInputVerlet {
        let x0 = self.x;
        let y0 = self.y;
        let mut x1 = self.vx;
        let mut y1 = self.vy;
        for (((x0, y0), x1), y1) in x0
            .iter()
            .zip(y0.iter())
            .zip(x1.iter_mut())
            .zip(y1.iter_mut())
        {
            *x1 = (x0 + *x1 * DT).rem_euclid(0.999);
            *y1 = (y0 + *y1 * DT).rem_euclid(0.999);
        }
        SimulationInputVerlet {
            x0,
            y0,
            x1,
            y1,
            k: self.k,
            relation_table: self.relation_table,
        }
    }
}

macro_rules! index_wrap {
    ($name:ident, $index:ty, $doc:expr) => {
        #[doc = $doc]
        #[repr(transparent)]
        #[derive(Copy, Clone, Default, Debug)]
        struct $name($index);
        unsafe impl bytemuck::Zeroable for $name {}
        unsafe impl bytemuck::Pod for $name {}
        impl From<$name> for usize {
            fn from(value: $name) -> usize {
                value.0 as _
            }
        }
        impl From<usize> for $name {
            fn from(value: usize) -> $name {
                $name(value as _)
            }
        }
        impl AddAssign<$name> for $name {
            fn add_assign(&mut self, other: Self) {
                self.0 += other.0;
            }
        }
        impl Add<$name> for $name {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }
    };
}
index_wrap!(Pid, u16, "Point index");
index_wrap!(Cid, u16, "Cell index");
#[derive(Debug)]
struct Tb<I, T, const SIZE: usize>(Box<[T; SIZE]>, PhantomData<I>);
impl<I, T, const SIZE: usize> From<Box<[T; SIZE]>> for Tb<I, T, SIZE> {
    fn from(value: Box<[T; SIZE]>) -> Tb<I, T, SIZE> {
        Tb(value, PhantomData)
    }
}

impl<I, T, const SIZE: usize> Deref for Tb<I, T, SIZE> {
    type Target = Box<[T; SIZE]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<I, T, const SIZE: usize> DerefMut for Tb<I, T, SIZE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl<I: Into<usize>, T, const SIZE: usize> Index<I> for Tb<I, T, SIZE> {
    type Output = T;
    fn index(&self, index: I) -> &Self::Output {
        &self.0[index.into()]
    }
}
impl<I: Into<usize>, T, const SIZE: usize> IndexMut<I> for Tb<I, T, SIZE> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index.into()]
    }
}
type Pbox<T> = Tb<Pid, T, NUM_POINTS_U>;
type Cbox<T> = Tb<Cid, T, CELLS_PLUS_1_U>;

/// Entire state of the simulation
struct SimulationState {
    x0: Pbox<f32>,
    y0: Pbox<f32>,
    x1: Pbox<f32>,
    y1: Pbox<f32>,
    k: Pbox<u8>,

    x0_tmp: Pbox<f32>,
    y0_tmp: Pbox<f32>,
    x1_tmp: Pbox<f32>,
    y1_tmp: Pbox<f32>,
    k_tmp: Pbox<u8>,

    surround_buffer_x: Vec<f32>,
    surround_buffer_y: Vec<f32>,
    surround_buffer_k: Vec<u8>,

    point_cell: Pbox<Cid>,

    point_cell_offset: Pbox<Pid>,

    
    cell_count: Cbox<Pid>,
    cell_index: Cbox<Pid>,
    //cell_index_end: Cbox<Pid>,


    relation_table: Box<[f32; KINDS2]>,
}
#[repr(C)]
#[derive(Copy, Clone)]
struct GpuPoint {
    x: f32,
    y: f32,
    k: u32,
    _unused: u32,
}
unsafe impl bytemuck::Zeroable for GpuPoint {}
unsafe impl bytemuck::Pod for GpuPoint {}

fn calloc<T: Default, const SIZE: usize, O>() -> O
where
    O: From<Box<[T; SIZE]>>,
{
    O::from(unsafe {
        (0..SIZE)
            .map(|_| T::default())
            .collect::<Vec<T>>()
            .try_into()
            .unwrap_unchecked()
    })
}
impl SimulationState {
    fn new(input: SimulationInputVerlet) -> Self {
        let SimulationInputVerlet {
            x0,
            y0,
            x1,
            y1,
            k,
            relation_table,
        } = input;
        let mut this = Self {
            x0: calloc(),
            y0: calloc(),
            x1: calloc(),
            y1: calloc(),
            k: calloc(),
            x0_tmp: x0,
            y0_tmp: y0,
            x1_tmp: x1,
            y1_tmp: y1,
            k_tmp: k,
            surround_buffer_x: Vec::new(),
            surround_buffer_y: Vec::new(),
            surround_buffer_k: Vec::new(),
            point_cell: calloc(),
            point_cell_offset: calloc(),
            cell_count: calloc(),
            cell_index: calloc(),
            relation_table,
        };
        this.cell_count.iter_mut().for_each(|e| *e = 0.into());
        for i in (0..NUM_POINTS_U).map(Into::into) {
            let new_x = this.x1_tmp[i];
            let new_y = this.y1_tmp[i];
            let cell = compute_cell(new_x, new_y);
            this.point_cell[i] = cell;
            this.point_cell_offset[i] = this.cell_count[cell];
            this.cell_count[cell] += 1.into();
        }
        let mut acc: Pid = 0.into();
        for i in (0..CELLS_PLUS_1_U).map(Into::into) {
            this.cell_index[i] = acc;
            let count = this.cell_count[i];
            acc += count;
        }
        for i in (0..NUM_POINTS_U).map(Into::into) {
            let cell = this.point_cell[i];
            let index = this.cell_index[cell] + this.point_cell_offset[i];
            this.x0[index] = this.x0_tmp[i];
            this.y0[index] = this.y0_tmp[i];
            this.x1[index] = this.x1_tmp[i];
            this.y1[index] = this.y1_tmp[i];
            this.k[index] = this.k_tmp[i];
        }
        this
    }
    #[inline(never)]
    // update starts right after previous physics update to reduce construction size.
    fn update(&mut self) {
        for cx in 0..CELLS_X {
            for cy in 0..CELLS_Y {
                //for i in (0..NUM_POINTS_U).map(Into::into) {
                let cell: Cid = usize::from(cx + cy * CELLS_X).into();
                self.surround_buffer_x.clear();
                self.surround_buffer_y.clear();
                self.surround_buffer_k.clear();
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let cell_x = cx as i32 + dx as i32;
                        let cell_y = cy as i32 + dy as i32;
                        let (offset_x, cell_x) = if cell_x == -1 {
                            (-1.0, CELLS_X as i32 - 1)
                        } else if cell_x == CELLS_X as i32 {
                            (1.0, 0)
                        } else {
                            (0.0, cell_x)
                        };
                        let (offset_y, cell_y) = if cell_y == -1 {
                            (-1.0, CELLS_Y as i32 - 1)
                        } else if cell_y == CELLS_Y as i32 {
                            (1.0, 0)
                        } else {
                            (0.0, cell_y)
                        };
                        let cell: Cid = ((cell_x + cell_y * CELLS_X as i32) as usize).into();

                        let range = usize::from(self.cell_index[cell])
                            ..usize::from(self.cell_index[cell+1.into()]);
                        for i in range.map(Into::into) {
                            self.surround_buffer_x.push(self.x1[i] + offset_x);
                            self.surround_buffer_y.push(self.y1[i] + offset_y);
                            self.surround_buffer_k.push(self.k[i]);
                        }
                    }
                }
                for i in (usize::from(self.cell_index[cell])
                    ..usize::from(self.cell_index[cell+1.into()]))
                    .map(Into::into)
                {
                    let x0 = self.x0[i];
                    let y0 = self.y0[i];
                    let x1 = self.x1[i];
                    let y1 = self.y1[i];
                    let k = self.k[i];
                    let mut acc_x = 0.0;
                    let mut acc_y = 0.0;
                    for ((&x_other, &y_other), k_other) in self
                        .surround_buffer_x
                        .iter()
                        .zip(self.surround_buffer_y.iter())
                        .zip(self.surround_buffer_k.iter())
                    {
                        // https://www.desmos.com/calculator/yos22615bv
                        //

                        let dx = x1 - x_other;
                        let dy = y1 - y_other;
                        let dot = dx * dx + dy * dy;
                        let r = dot.sqrt();

                        const BETA: f32 = 0.3;
                        let f = {
                            let x = r * (1.0 / POINT_MAX_RADIUS);
                            let k: f32 =
                                0.2 * self.relation_table[(k_other + k * NUM_KINDS as u8) as usize];
                            //let c = 1.0 - x / BETA;
                            let c = 0.1 * (1.0 / (x * x) - 1.0 / (BETA * BETA));
                            ((2.0 * k / (1.0 - BETA)) * (x - BETA).min(1.0 - x).max(0.0)).max(c)
                        };
                        let dir_x = dx / r;
                        let dir_y = dy / r;
                        //let f: f32 = 0.001
                        //    * (1.0 / (r * r) - 1.0 / (POINT_MAX_RADIUS * POINT_MAX_RADIUS))
                        //        .max(0.0);
                        if f.is_finite() && dir_x.is_finite() && dir_y.is_finite() {
                            acc_x += dir_x * f;
                            acc_y += dir_y * f;
                        }
                    }
                    let mut vx = x1 - x0;
                    let mut vy = y1 - y0;

                    for dx in [-1.0, 0.0, 1.0] {
                        let c = x1 - x0 + dx;
                        if c.abs() < vx.abs() {
                            vx = c
                        }
                    }
                    for dy in [-1.0, 0.0, 1.0] {
                        let c = y1 - y0 + dy;
                        if c.abs() < vy.abs() {
                            vy = c
                        }
                    }
                    let v2 = vx * vx + vy * vy;
                    let vx_norm = vx / v2.sqrt();
                    let vy_norm = vy / v2.sqrt();
                    let friction_force = v2 * (100.0 / (DT * DT));
                    acc_x += friction_force * (-vx_norm);
                    acc_y += friction_force * (-vy_norm);

                    let new_x = (x1 + vx + acc_x * DT * DT).rem_euclid(0.99999);
                    let new_y = (y1 + vy + acc_y * DT * DT).rem_euclid(0.99999);
                    self.x0[i] = x1;
                    self.x1[i] = new_x;
                    self.y0[i] = y1;
                    self.y1[i] = new_y;

                    self.x0_tmp[i] = self.x0[i];
                    self.y0_tmp[i] = self.y0[i];
                    self.x1_tmp[i] = self.x1[i];
                    self.y1_tmp[i] = self.y1[i];
                    self.k_tmp[i] = k;
                }
            }
        }
        self.cell_count.iter_mut().for_each(|e| *e = 0.into());
        for i in (0..NUM_POINTS_U).map(Into::into) {
            let new_x = self.x1_tmp[i];
            let new_y = self.y1_tmp[i];
            let cell = compute_cell(new_x, new_y);
            self.point_cell[i] = cell;
            self.point_cell_offset[i] = self.cell_count[cell];
            self.cell_count[cell] += 1.into();
        }
        let mut acc: Pid = 0.into();
        for i in (0..CELLS_PLUS_1_U).map(Into::into) {
            self.cell_index[i] = acc;
            let count = self.cell_count[i];
            acc += count;
        }
        for i in (0..NUM_POINTS_U).map(Into::into) {
            let cell = self.point_cell[i];
            let index = self.cell_index[cell] + self.point_cell_offset[i];
            self.x0[index] = self.x0_tmp[i];
            self.y0[index] = self.y0_tmp[i];
            self.x1[index] = self.x1_tmp[i];
            self.y1[index] = self.y1_tmp[i];
            self.k[index] = self.k_tmp[i];
        }
        /*
        // PPA
        ppa_offset_1(&mut self.cell_count_or_index_new);
        // Insertion.
        {
            for i_old in (0..NUM_POINTS_U).map(Into::into) {
                let i_new = self.cell_count_or_index_new[self.point_cell_location[i_old]]
                    + self.point_cell_offset[i_old];
                self.x0[i_new] = self.x0_tmp[i_old];
                self.y0[i_new] = self.y0_tmp[i_old];
                //self.k_new[i_new] = self.k[i_old];
                //self.back[i_new] = i_old as _;
            }
        };
        // Swapping vectors.
        swap(&mut self.x0, &mut self.x1);
        swap(&mut self.y0, &mut self.y1);
        //swap(&mut self.k, &mut self.k_new);
        swap(
            &mut self.cell_count_or_index,
            &mut self.cell_count_or_index_new,
        );

        // <- at this point state is valid

        self.cell_count_or_index_new
            .iter_mut()
            .for_each(|e| *e = 0.into());
        for cell_y in 0..CELLS_Y {
            for cell_x in 0..CELLS_X {
                let cell = Cid(cell_y * CELLS_Y + cell_x);
                self.surround_buffer_x.clear();
                self.surround_buffer_y.clear();
                self.surround_buffer_k.clear();
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let cell = ((((cell_y as i32 + dy + CELLS_Y as i32) % CELLS_Y as i32)
                            * CELLS_X as i32
                            + ((cell_x as i32 + dx + CELLS_X as i32) % CELLS_X as i32))
                            as usize)
                            .into();
                        for i in (usize::from(self.cell_count_or_index[cell])
                            ..usize::from(self.cell_count_or_index[cell + 1.into()]))
                            .map(Into::into)
                        {
                            self.surround_buffer_x.push(self.x1[i]);
                            self.surround_buffer_y.push(self.y1[i]);
                            //self.surround_buffer_k.push(self.k[i]);
                        }
                    }
                }
                for i in (usize::from(self.cell_count_or_index[cell])
                    ..usize::from(self.cell_count_or_index[cell + 1.into()]))
                    .map(Into::into)
                {
                    let x = self.x1[i];
                    let y = self.y1[i];
                    //let k = self.k_new[i];
                    let x_prev: f32 = todo!();//self.x0[self.back[i]];
                    let y_prev: f32 = todo!();//self.y0[self.back[i]];
                    let mut acc_x = 0.0;
                    let mut acc_y = 0.0;
                    for ((other_x, other_y), &other_k) in self
                        .surround_buffer_x
                        .iter()
                        .zip(self.surround_buffer_y.iter())
                        .zip(self.surround_buffer_k.iter())
                    {
                        let diff_x = x - other_x;
                        let diff_y = y - other_y;
                        let r2 = diff_x * diff_x + diff_y * diff_y;

                        let invr = 1.0 / r2.sqrt();

                        let dx = diff_x * invr;
                        let dy = diff_y * invr;

                        let force = 1.0//self.relation_table
                        //[(k as u16 + NUM_KINDS as u16 * other_k as u16) as usize]
                        * 0.01 // TODO: move DT, this factor into relation table.
                        * invr
                        * invr;

                        if dx.is_finite() && dy.is_finite() && force.is_finite() {
                            acc_x += dx * force;
                            acc_y += dy * force;
                        }
                    }
                    if !acc_x.is_finite() {
                        acc_x = 0.0;
                    }
                    if !acc_y.is_finite() {
                        acc_x = 0.0;
                    }
                    let x_next = x_prev.rem_euclid(0.999); //(x+acc_x*0.00001).rem_euclid(0.9999);//let x_next = (x + (x - x_prev) * 0.95 + acc_x * DT * DT).rem_euclid(0.9999);
                    let y_next = y_prev.rem_euclid(0.999); //(y+acc_y*0.00001).rem_euclid(0.9999);//let y_next = (y + (y - y_prev) * 0.95 + acc_y * DT * DT).rem_euclid(0.9999);

                    {
                        let new_cell = compute_cell(x_next, y_next);
                        self.point_cell_location[i] = new_cell;
                        let new_offset = self.cell_count_or_index_new[new_cell];
                        self.cell_count_or_index_new[new_cell] += 1.into();
                        self.point_cell_offset[i] = new_offset;
                        self.x0_tmp[i] = x_next;
                        self.y0_tmp[i] = y_next;
                    };
                }
            }
        }
        */
    }
    fn serialize_gpu(&self, data: &mut Vec<GpuPoint>) {
        data.clear();
        data.extend(
            self.x0
                .iter()
                .zip(self.y0.iter())
                .zip(self.k.iter())
                .map(|((&x, &y), &k)| GpuPoint {
                    x,
                    y,
                    k: k as _,
                    _unused: 0,
                }),
        )
    }
}

/// PPA and offset by 1.
fn ppa_offset_1(cell_count_or_index_new: &mut Cbox<Pid>) {
    let mut acc: Pid = 0.into();
    for i in (0..CELLS_PLUS_1_U).map(Into::into) {
        let curr = cell_count_or_index_new[i];
        cell_count_or_index_new[i] = acc;
        acc += curr;
    }
}

macro_rules! printlnc {
    ($aa:expr, $bb:expr) => {
        #[cfg(target_arch = "wasm32")]
        {
            web_sys::console::log_1(&format!($aa, $bb).into());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            println!($aa, $bb);
        }
    };
}
macro_rules! dbgc {
    ($aa:expr) => {{
        match $aa {
            tmp => {
                eprintln!(
                    "[{}:{}] {} = {:#?}",
                    file!(),
                    line!(),
                    stringify!($aa),
                    &tmp
                );
                tmp
            }
        }
    }};
}

#[derive(Debug)]
struct SimParams {
    forces: Vec<f32>,
}

impl SimParams {
    fn new() -> SimParams {
        let mut rng;
        //rng = Prng(13);
        cfg_if::cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                rng = Prng(13);
            } else {
                rng = Prng(SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_micros() as u64);
            }
        }

        printlnc!("seed: {}", rng.0);

        SimParams {
            forces: (0..NUM_KINDS * NUM_KINDS)
                .map(|_| rng.f32(-1.0..1.0))
                .collect(),
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    let (event_loop, window) = window_setup();

    let mut state = State::new(window).await;
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, window_id } if window_id == state.window.id() => match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(VirtualKeyCode::Escape),
                        ..
                    },
                ..
            } => *control_flow = ControlFlow::Exit,
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                use VirtualKeyCode::*;
                match keycode {
                    U => state.use_simd = dbgc!(!state.use_simd),
                    _ => (),
                }
            }
            WindowEvent::Resized(physical_size) => state.resize(physical_size),
            WindowEvent::ScaleFactorChanged { new_inner_size, .. } => state.resize(*new_inner_size),
            _ => state.input(&event),
        },
        Event::RedrawRequested(window_id) if window_id == state.window().id() => {
            state.update();
            match state.render() {
                Ok(_) => {}
                Err(SurfaceError::Lost) => state.resize(state.size),
                Err(SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            state.window().request_redraw();
        }
        _ => {}
    });
}

fn window_setup() -> (EventLoop<()>, Window) {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            env_logger::init();
        }
    }

    let event_loop = EventLoop::new();

    let window_size = winit::dpi::LogicalSize {
        width: 512,
        height: 512,
    };
    let window = WindowBuilder::new()
        .with_inner_size(window_size)
        .with_max_inner_size(window_size)
        .with_min_inner_size(window_size)
        .build(&event_loop)
        .unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        use winit::dpi::PhysicalSize;
        window.set_inner_size(PhysicalSize::new(450, 400));

        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("id-thingy")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }
    (event_loop, window)
}
struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,

    render_pipeline: wgpu::RenderPipeline,

    bind_group: wgpu::BindGroup,

    #[cfg(not(target_arch = "wasm32"))]
    last_print: Instant,
    frame: usize,

    use_simd: bool,

    simulation_state: SimulationState,

    point_transfer_buffer: Vec<GpuPoint>,
    gpu_point_buffer: wgpu::Buffer,
}

macro_rules! timed {
    ($name:expr, $code:block) => {{
        static mut COUNT: u8 = 0;
        static mut TIME: Duration = Duration::ZERO;
        if unsafe {
            COUNT += 1;
            COUNT == 120
        } {
            println!(
                "{}: {:?} potential fps, {:?} seconds/update",
                $name,
                120.0 / unsafe { TIME }.as_secs_f64(),
                unsafe { TIME }.as_secs_f64() / 120.0
            );
            unsafe {
                TIME = Duration::ZERO;
                COUNT = 0;
            }
        }
        let _time_pre: Instant = Instant::now();
        {
            $code
        }
        unsafe {
            TIME += Instant::now().duration_since(_time_pre);
        }
    }};
}

impl State {
    async fn new(window: Window) -> Self {
        let simulation_state = SimulationState::new(SimulationInputEuler::new().verlet());

        let size: winit::dpi::PhysicalSize<u32> = window.inner_size();
        let instance: wgpu::Instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });
        dbgc!(instance
            .enumerate_adapters(wgpu::Backends::all())
            .collect::<Vec<_>>());
        let surface: wgpu::Surface = unsafe { instance.create_surface(&window) }.unwrap();
        let adapter: wgpu::Adapter = instance
            .enumerate_adapters(wgpu::Backends::all())
            .find(|adapter| adapter.is_surface_supported(&surface))
            .unwrap();
        dbgc!(adapter.features());
        dbgc!(adapter.limits());

        let limits = wgpu::Limits::downlevel_defaults(); //downlevel_webgl2_defaults();
        
        dbgc!(&limits);
        println!("Expected points per cell: {}", NUM_POINTS as f32 / CELLS as f32);
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits,
                    label: None,
                },
                None,
            )
            .await
            .unwrap();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format: wgpu::TextureFormat = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            //present_mode: wgpu::PresentMode::Immediate,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let mut point_transfer_buffer = Vec::new();
        simulation_state.serialize_gpu(&mut point_transfer_buffer);
        let gpu_point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&point_transfer_buffer),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let bind_group_layout: wgpu::BindGroupLayout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Default::default(),
                    },
                    count: None,
                }],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(
                    gpu_point_buffer.as_entire_buffer_binding(),
                ),
            }],
        });

        let shader_source: String = std::fs::read_to_string("src/shader.wgsl")
            .unwrap()
            .replace("{NUM_POINTS}", &format!("{}", NUM_POINTS))
            .replace("{NUM_KINDS}", &format!("{}", NUM_KINDS));

        let render_pipeline;
        {
            let fragment_vertex_shader: wgpu::ShaderModule =
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Fragment and vertex shader"),
                    source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                });
            let pipeline_layout: wgpu::PipelineLayout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });
            let layout = Some(&pipeline_layout);
            let make_primitive = |topology| wgpu::PrimitiveState {
                topology,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            };
            let multisample: wgpu::MultisampleState = wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            };

            render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout,
                vertex: wgpu::VertexState {
                    module: &fragment_vertex_shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fragment_vertex_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: make_primitive(wgpu::PrimitiveTopology::TriangleList),
                depth_stencil: None,
                multisample,
                multiview: None,
            });
        }

        Self {
            surface,
            device,
            queue,
            config,
            size,
            window,
            render_pipeline,
            bind_group,
            #[cfg(not(target_arch = "wasm32"))]
            last_print: Instant::now(),
            frame: 0,
            use_simd: true,
            simulation_state,
            point_transfer_buffer,
            gpu_point_buffer,
        }
    }

    fn render(&mut self) -> Result<(), SurfaceError> {
        self.frame += 1;
        let max_frame = 240;
        #[cfg(not(target_arch = "wasm32"))]
        if self.frame == max_frame {
            self.frame = 0;
            let dt = replace(&mut self.last_print, Instant::now()).elapsed();
            let fps = max_frame as f64 / dt.as_secs_f64();
            printlnc!("fps: {fps:?}, dt: {:50?}", dt);
        }

        let output: wgpu::SurfaceTexture = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        timed!("Update with overhead", {
            self.queue.write_buffer(
                &self.gpu_point_buffer,
                0,
                bytemuck::cast_slice(&self.point_transfer_buffer),
            );
            timed!("Core update", {
                for _ in 0..3 {
                    self.simulation_state.update();
                }
            });
            self.simulation_state
                .serialize_gpu(&mut self.point_transfer_buffer);
        });
        let mut encoder: wgpu::CommandEncoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Encoder"),
                });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..(3 * NUM_POINTS), 0..1);
        drop(render_pass);

        self.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }

    fn update(&mut self) {}

    fn input(&mut self, _event: &WindowEvent<'_>) {}

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }
}

use core::arch::x86_64::*;
use simd::*;
mod simd {
    use core::arch::x86_64::*;
    #[inline(always)]
    pub unsafe fn mul(a: __m256, b: __m256) -> __m256 {
        _mm256_mul_ps(a, b)
    }
    #[inline(always)]
    pub unsafe fn add(a: __m256, b: __m256) -> __m256 {
        _mm256_add_ps(a, b)
    }
    #[inline(always)]
    pub unsafe fn andi(a: __m256i, b: __m256i) -> __m256i {
        _mm256_and_si256(a, b)
    }
    #[inline(always)]
    pub unsafe fn andn(a: __m256, b: __m256) -> __m256 {
        _mm256_andnot_ps(a, b)
    }
    #[inline(always)]
    pub unsafe fn sub(a: __m256, b: __m256) -> __m256 {
        _mm256_sub_ps(a, b)
    }
    #[inline(always)]
    pub unsafe fn fmadd(a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_fmadd_ps(a, b, c)
    }
    #[inline(always)]
    pub unsafe fn gather<const SCALE: i32>(p: *const f32, indexes: __m256i) -> __m256 {
        _mm256_i32gather_ps::<SCALE>(p, indexes)
    }
    #[inline(always)]
    pub unsafe fn rsqrt(a: __m256) -> __m256 {
        _mm256_rsqrt_ps(a)
    }
    #[inline(always)]
    pub unsafe fn splat(a: f32) -> __m256 {
        _mm256_set1_ps(a)
    }
    #[inline(always)]
    pub unsafe fn splati(a: i32) -> __m256i {
        _mm256_set1_epi32(a)
    }
    #[inline(always)]
    pub unsafe fn max(a: __m256, b: __m256) -> __m256 {
        _mm256_max_ps(a, b)
    }
    #[inline(always)]
    pub unsafe fn min(a: __m256, b: __m256) -> __m256 {
        _mm256_min_ps(a, b)
    }
    #[inline(always)]
    pub unsafe fn slli<const IMM: i32>(a: __m256i) -> __m256i {
        _mm256_slli_epi32::<IMM>(a)
    }

    pub fn as_rchunks<T, const N: usize>(this: &[T]) -> (&[T], &[[T; N]]) {
        unsafe fn as_chunks_unchecked<T, const N: usize>(s: &[T]) -> &[[T; N]] {
            let new_len = s.len() / N;
            unsafe { std::slice::from_raw_parts(s.as_ptr().cast(), new_len) }
        }
        assert!(N != 0, "chunk size must be non-zero");
        let len = this.len() / N;
        let (remainder, multiple_of_n) = this.split_at(this.len() - len * N);
        // SAFETY: We already panicked for zero, and ensured by construction
        // that the length of the subslice is a multiple of N.
        let array_slice = unsafe { as_chunks_unchecked(multiple_of_n) };
        (remainder, array_slice)
    }
}
