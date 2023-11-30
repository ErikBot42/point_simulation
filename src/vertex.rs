// use std::mem::size_of;
// #[repr(C)]
// #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
// pub struct Vertex {
//     position: [f32; 2],
// }
// impl Vertex {
//     const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x2];
//     pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
//         wgpu::VertexBufferLayout {
//             array_stride: size_of::<Self>() as wgpu::BufferAddress,
//             step_mode: wgpu::VertexStepMode::Vertex,
//             attributes: &Self::ATTRIBS,
//         }
//     }
// }
// 
// // A    D
// //
// // C    B
// #[rustfmt::skip]
// pub const VERTICES: &[Vertex] = {
//     const A: Vertex = Vertex { position: [ 1.0,  1.0]};
//     const B: Vertex = Vertex { position: [-1.0, -1.0]};
//     const C: Vertex = Vertex { position: [ 1.0, -1.0]};
//     const D: Vertex = Vertex { position: [-1.0,  1.0]};
//     &[
//         A, B, C, 
//         A, D, B,
//     ]
// };
// 
