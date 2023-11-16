
// @builtin(local_invocation_id) LocalInvocationID: vec3<u32>, // position in workgroup
// @builtin(local_invocation_index) LocalInvocationIndex: u32, // same but linearized
// @builtin(global_invocation_id) GlobalInvocationID: vec3<u32>, // position in compute shader grid
// @builtin(workgroup_id) WorkgroupID: vec3<u32>, // what workgroup this is
// @builtin(num_workgroups) NumWorkgroups: vec3<u32>, // dispatch size
struct GpuPoint {
x: f32,
       y: f32,
       k: u32,
       _unused: u32,
}

@compute @workgroup_size(WORKGROUP_SIZE)
    fn serialize(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {

        let i = GlobalInvocationID.x;
        var p: GpuPoint;
        p.x = x0[i];
        p.y = y0[i];
        p.k = k[i];
        p._unused = 0u;
        points[i] = p;
    }

@compute @workgroup_size(WORKGROUP_SIZE)
    fn update(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {

        let i = GlobalInvocationID.x;

        let cell: u32 = point_cell[i];
        // x-range is max(0, cell_x-1)..min(CELLS_X-1, cell_x+1)


        let cell_x: u32 = cell % CELLS_Xu;
        let cell_y: u32 = cell / CELLS_Xu;

        let cell_x_min = u32(max(0, i32(cell_x) - 1));
        let cell_x_max = u32(min(i32(CELLS_X) - 1, i32(cell_x) + 1));

        let cell_y_min = u32(max(0, i32(cell_y) - 1));
        let cell_y_max = u32(min(i32(CELLS_Y) - 1, i32(cell_y) + 1));

        let my_x0 = x0[i];
        let my_x1 = x1[i];

        let my_y0 = y0[i];
        let my_y1 = y1[i];

        let my_k = k[i];

        var acc_x: f32 = 0.0;
        var acc_y: f32 = 0.0;

        for (var cy: u32 = cell_y_min; cy <= cell_y_max; cy+=1u) {
            let index_min = cell_index[u32(cy * CELLS_Xu + cell_x_min)];
            let index_max = cell_index[u32(cy * CELLS_Xu + cell_x_max + 1u)];
            for (var index: u32 = index_min; index < index_max; index += 1u) {
                if (index == i) { continue; }

                let x_other: f32 = x1[index];
                let y_other: f32 = y1[index];
                let k_other: u32 = k[index];

                // TODO: FMA

                let dx = my_x1 - x_other;
                let dy = my_y1 - y_other;

                let ddot = dx * dx + dy * dy;
                let r = sqrt(ddot);

                let x = r * (1.0 / POINT_MAX_RADIUS);

                let c = 4.0 * (1.0 - x / BETA);

                let k_fac = relation_table[my_k + k_other * NUM_KINDSu]; // TODO

                let f = max(
                        (k_fac / (1.0 - BETA)) * max(min(x - BETA, 1.0 - x), 0.0)
                        , c);

                let dir_x = dx / r;
                let dir_y = dy / r;

                if ddot != 0.0 {
                    acc_x += dir_x * f;
                    acc_y += dir_y * f;
                }
            }
        }
        acc_x *= MIN_WIDTH.0 * 100.0;
        acc_y *= MIN_WIDTH.0 * 100.0;

        var vx = my_x1 - my_x0;
        var vy = my_y1 - my_y0;

        if (my_x0 > 1.0) {
            vx = - abs(vx);
        }
        if (my_x0 < 0.0) {
            vx = abs(vx);
        }
        if (my_y0 > 1.0) {
            vy = - abs(vy);
        }
        if (my_y0 < 0.0) {
            vy = abs(vy);
        }

        let v2 = vx * vx + vy * vy;
        let vx_norm = vx / sqrt(v2);
        let vy_norm = vy / sqrt(v2);

        let friction_force = v2 * (50.0 / (DT * DT));



        acc_x += friction_force * (-vx_norm);
        acc_y += friction_force * (-vy_norm);

        // TODO: detect NaN

        let new_x = my_x1 + vx + acc_x * DT * DT;
        let new_y = my_y1 + vy + acc_y * DT * DT;

        x1_tmp[i] = new_x;
        y1_tmp[i] = new_y;

        let new_cell = compute_cell(new_x, new_y);
        point_cell[i] = new_cell;
        point_cell_offset[i] = atomicAdd(&cell_count[i], 1u);
    }


var<workgroup> cache0: array<u32, 256>;
var<workgroup> cache1: array<u32, 256>;
// NOTE: needs padding because exclusive prefix add.

// copy data and perform ppa with granularity of 256
// strategy: cut of work to tid < N, so that empty subgroups finish quickly.
// dispatch this N/256 (+1?) times
@compute @workgroup_size(256)
    fn exclusive_ppa256(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
        let i: u32 = GlobalInvocationID.x;
        let i_cache: u32 = i & 0xFFu; // 256
        if (i == 0u) {

        }
        cache0[i_cache] = cell_count[i];
        cell_count[i] = 0u; // maybe UB?

        // simple but work-inefficient
        workgroupBarrier();
        if (i_cache > 1u)   { cache1[i_cache] = cache0[i_cache] + cache0[i_cache - 1u];  } else { cache1[i_cache] = cache0[i_cache]; } workgroupBarrier();
        if (i_cache > 2u)   { cache0[i_cache] = cache0[i_cache] + cache1[i_cache - 2u];  } else { cache0[i_cache] = cache1[i_cache]; } workgroupBarrier();
        if (i_cache > 4u)   { cache1[i_cache] = cache0[i_cache] + cache0[i_cache - 4u];  } else { cache1[i_cache] = cache0[i_cache]; } workgroupBarrier();
        if (i_cache > 8u)   { cache0[i_cache] = cache0[i_cache] + cache1[i_cache - 8u];  } else { cache0[i_cache] = cache1[i_cache]; } workgroupBarrier();
        if (i_cache > 16u)  { cache1[i_cache] = cache0[i_cache] + cache0[i_cache - 16u]; } else { cache1[i_cache] = cache0[i_cache]; } workgroupBarrier();
        if (i_cache > 32u)  { cache0[i_cache] = cache0[i_cache] + cache1[i_cache - 32u]; } else { cache0[i_cache] = cache1[i_cache]; } workgroupBarrier();
        if (i_cache > 64u)  { cache1[i_cache] = cache0[i_cache] + cache0[i_cache - 64u]; } else { cache1[i_cache] = cache0[i_cache]; } workgroupBarrier();
        if (i_cache > 128u) { 
            point_cell_offset[i+1u] = cache0[i_cache] + cache1[i_cache - 128u]; 
        } else {
            point_cell_offset[i+1u] = cache0[i_cache];
        }
        // the following is done automatically from initialization:
        // if (i == 0u) {
        //     point_cell_offset[0] = 0u;
        // }
    }

@compute @workgroup_size(256)
    fn full_ppa(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
        // perform ppa again, but on the "high" elements of each vector.

        // NOTE: this has a limit



        // increment batches based on meta ppa results

        let i: u32 = GlobalInvocationID.x;
        let i_cache: u32 = i & 0xFFu; // = i if things are going as planned

        if (i_cache * 256u < (CELLSu + 1u)) {
            cache0[i_cache] = point_cell_offset[i_cache * 256u + 255u];
        } else {
            cache0[i_cache] = 0u;
        }
        
        // we support 256 * 256 cells, probably?
        workgroupBarrier();
        if (i_cache > 1u)   { cache1[i_cache] = cache0[i_cache] + cache0[i_cache - 1u];   } else { cache1[i_cache] = cache0[i_cache]; } workgroupBarrier();
        if (i_cache > 2u)   { cache0[i_cache] = cache1[i_cache] + cache1[i_cache - 2u];   } else { cache0[i_cache] = cache1[i_cache]; } workgroupBarrier();
        if (i_cache > 4u)   { cache1[i_cache] = cache0[i_cache] + cache0[i_cache - 4u];   } else { cache1[i_cache] = cache0[i_cache]; } workgroupBarrier();
        if (i_cache > 8u)   { cache0[i_cache] = cache1[i_cache] + cache1[i_cache - 8u];   } else { cache0[i_cache] = cache1[i_cache]; } workgroupBarrier();
        if (i_cache > 16u)  { cache1[i_cache] = cache0[i_cache] + cache0[i_cache - 16u];  } else { cache1[i_cache] = cache0[i_cache]; } workgroupBarrier();
        if (i_cache > 32u)  { cache0[i_cache] = cache1[i_cache] + cache1[i_cache - 32u];  } else { cache0[i_cache] = cache1[i_cache]; } workgroupBarrier();
        if (i_cache > 64u)  { cache1[i_cache] = cache0[i_cache] + cache0[i_cache - 64u];  } else { cache1[i_cache] = cache0[i_cache]; } workgroupBarrier();

        var local_offset: u32; 
        if (i_cache > 128u) { 
            local_offset = cache1[i_cache] + cache1[i_cache - 128u]; 
        } else { 
            local_offset = cache1[i_cache]; 
        } 

        let i_start: u32 = i * 256u + 256u;
        for (var ii: u32 = i_start; ii < min(i_start + 256u, CELLSu + 1u); ii++) {
            point_cell_offset[ii] += local_offset;
        }
    }


fn compute_cell(xi: f32, yi: f32) -> u32 {
    let x: f32 = inv_scale_simulation_x(xi);
    let y: f32 = inv_scale_simulation_y(yi);
    // x in 0_f32..1_f32
    // y in 0_f32..1_f32

    let cell_x = min(u32(x * CELLS_X.0), CELLS_Xu - 1u);
    let cell_y = min(u32(y * CELLS_Y.0), CELLS_Yu - 1u);
    let cell = cell_x + cell_y * CELLS_Xu;
    return cell;
}
fn inv_scale_simulation_x(f: f32) -> f32 {
    return (f - MIN_X) / (MAX_X - MIN_X);
}
fn inv_scale_simulation_y(f: f32) -> f32 {
    return (f - MIN_Y) / (MAX_Y - MIN_Y);
}
