use egui::epaint::{ClippedShape, Primitive};
use egui::{ClippedPrimitive, FontImage, ImageData, Shape, TexturesDelta};
use grr::{BaseFormat, BlendChannel, BlendFactor, BlendOp, ColorBlend, ColorBlendAttachment, Constant, Extent, Filter, Format, FormatLayout, HostImageCopy, ImageType, MemoryFlags, MemoryLayout, Offset, Region, SamplerAddress, SamplerDesc, ShaderFlags, ShaderSource, SubresourceLayers, VertexAttributeDesc, VertexBufferView, Viewport};
use {
    ahash::AHashMap,
    egui::{emath::Rect, epaint::Mesh, epaint::TextureId},
};


const VERTEX_SRC : &str = r#"#version 460 core

layout (location = 0) uniform vec2 u_screen_size;

layout (location = 0) in vec2 v_pos;
layout (location = 1) in vec2 v_tc;
layout (location = 2) in uvec4 v_srgba;



layout (location = 0) out vec4 a_srgba;
layout (location = 1) out vec2 a_tc;

// 0-1 linear  from  0-255 sRGB
vec3 linear_from_srgb(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(10.31475));
    vec3 lower = srgb / vec3(3294.6);
    vec3 higher = pow((srgb + vec3(14.025)) / vec3(269.025), vec3(2.4));
    return mix(higher, lower, vec3(cutoff));
}

vec4 linear_from_srgba(vec4 srgba) {
    return vec4(linear_from_srgb(srgba.rgb), srgba.a / 255.0);
}

void main() {
    gl_Position = vec4(
                      2.0 * v_pos.x / u_screen_size.x - 1.0,
                      1.0 - 2.0 * v_pos.y / u_screen_size.y,
                      0.0,
                      1.0);
    // egui encodes vertex colors in gamma spaces, so we must decode the colors here:
    a_srgba = linear_from_srgba(v_srgba);
    a_tc = v_tc;
}"#;

const FRAGMENT_SRC : &str = r#"#version 460 core

layout (binding = 0) uniform sampler2D u_sampler;

layout (location = 0) in vec4 a_rgba;
layout (location = 1) in vec2 a_tc;

out vec4 frag_color;

void main() {
// The texture is set up with `SRGB8_ALPHA8`, so no need to decode here!
vec4 texture_rgba = texture(u_sampler, a_tc);
/// Multiply vertex color with texture color (in linear space).
frag_color = a_rgba * texture_rgba;
}"#;

pub struct Painter {
    max_texture_side: usize,
    ibo : grr::Buffer,
    vbo : grr::Buffer,
    sampler : grr::Sampler,
    pipeline: grr::Pipeline,
    vertex_array: grr::VertexArray,

    textures: AHashMap<egui::TextureId, (grr::ImageView, grr::Image)>,
}
#[derive(Copy, Clone, Debug)]
pub struct PainterSettings{
    pub ibo_size : u64,
    pub vbo_size : u64,
    pub max_texture_side: usize
}

impl std::default::Default for PainterSettings {
    fn default() -> Self {
        Self{
            ibo_size: 10048,
            vbo_size: 10096,
            max_texture_side: 4096,
        }
    }
}

impl Painter {
    pub fn new(device: &grr::Device, settings : PainterSettings) -> grr::Result<Painter> {
        let max_texture_side = settings.max_texture_side;

        let vertex_array = unsafe {
            let vao = device.create_vertex_array(&[VertexAttributeDesc {
                location: 0,
                binding: 0,
                format: grr::VertexFormat::Xy32Float,
                offset: 0,
            }, VertexAttributeDesc {
                location: 1,
                binding: 0,
                format: grr::VertexFormat::Xy32Float,
                offset: (2 * std::mem::size_of::<f32>()) as _,
            }, VertexAttributeDesc {
                location: 2,
                binding: 0,
                format: grr::VertexFormat::Xyzw8Uint,
                offset: (4 * std::mem::size_of::<f32>()) as _,
            }
            ])?;
            device.object_name(vao, "egui vao");
            vao
        };

        let vbo = unsafe{
            device.create_buffer(settings.vbo_size, MemoryFlags::DYNAMIC)
        }?;

        let ibo = unsafe {
            device.create_buffer(settings.ibo_size, MemoryFlags::DYNAMIC)
        }?;

        let vertex_shader = unsafe {
            let bytes = VERTEX_SRC.as_bytes();
            device.create_shader(grr::ShaderStage::Vertex, ShaderSource::Glsl, bytes, ShaderFlags::VERBOSE)
        }?;

        let fragment_shader = unsafe {
            let bytes = FRAGMENT_SRC.as_bytes();
            device.create_shader(grr::ShaderStage::Fragment, ShaderSource::Glsl, bytes, ShaderFlags::VERBOSE)
        }?;

        let pipeline = unsafe {
            device.create_graphics_pipeline(
                grr::GraphicsPipelineDesc {
                    vertex_shader: Some(vertex_shader),
                    tessellation_control_shader: None,
                    tessellation_evaluation_shader: None,
                    geometry_shader: None,
                    fragment_shader: Some(fragment_shader),
                    mesh_shader: None,
                    task_shader: None,
                },
                grr::PipelineFlags::VERBOSE,
            )?
        };
        let sampler = unsafe {
            device.create_sampler(SamplerDesc {
                min_filter: Filter::Linear,
                mag_filter: Filter::Linear,
                mip_map: None,
                address: (SamplerAddress::ClampEdge, SamplerAddress::ClampEdge, SamplerAddress::ClampEdge),
                lod_bias: 0.0,
                lod: 0.0..10.0,
                compare: None,
                border_color: [0.0, 0.0, 0.0, 1.0],
            })
        }?;

        unsafe {
            device.delete_shaders(&[vertex_shader, fragment_shader]);
        }

        Ok(Painter {
            ibo,
            vbo,
            sampler,
            max_texture_side,
            pipeline,
            vertex_array,
            textures: Default::default(),
        })
    }

    pub fn max_texture_side(&self) -> usize {
        self.max_texture_side
    }

    pub fn paint(
        &mut self,
        device: &grr::Device,
        pixels_per_point: f32,
        dimensions: [u32; 2],
        cipped_meshes: Vec<ClippedPrimitive>,
        textures_delta: TexturesDelta
    ) {
        self.update_textures(device, &textures_delta);
        for c in cipped_meshes {
            self.paint_mesh(device, pixels_per_point, c.clip_rect, dimensions, &c.primitive);
        }
        for i in textures_delta.free{
            unsafe { self.free_texture(i, device) };
        }
    }
    fn update_textures(&mut self, device : &grr::Device, textures : &TexturesDelta){

        for (id, data) in &textures.set{

            if let Some(pos) = data.pos{
                //TODO: update subregion
                return;
            }


            match &data.image{
                ImageData::Font(im) => {
                    self.set_font_texture(&device, id.clone(), &im).unwrap();
                }
                ImageData::Color(im) => {

                    let mut pixels = Vec::with_capacity(4 * im.pixels.len());
                    for pix in &im.pixels{
                        pixels.push(pix.r());
                        pixels.push(pix.g());
                        pixels.push(pix.b());
                        pixels.push(pix.a());
                    }

                    self.set_color_texture(&device, id.clone(), &pixels,im.size[0] as _, im.size[1] as _).unwrap();
                }
            }
        }
    }
    #[inline(never)] // Easier profiling
    fn paint_mesh(
        &mut self,
        device: &grr::Device,
        pixels_per_point: f32,
        clip_rect: Rect,
        dimensions: [u32; 2],
        mesh: &Primitive,
    ) {




        let mut opt = None;
        if let Primitive::Mesh(m) = mesh{
            opt = Some(m);
        }else{
            return;
        }
        let mesh = opt.unwrap();



        {
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct Vertex {
                a_pos: [f32; 2],
                a_tc: [f32; 2],
                a_srgba: [u8; 4],
            }

            let vertices: &[Vertex] = bytemuck::cast_slice(&mesh.vertices);

            unsafe { device.copy_host_to_buffer(self.vbo, 0, bytemuck::cast_slice(vertices))}
        };
        let vertex_buffer = self.vbo;
        let indices = &mesh.indices.len();
        let index_buffer = self.ibo;
        unsafe { device.copy_host_to_buffer(self.ibo, 0, bytemuck::cast_slice(&mesh.indices)) }
        let (width_in_pixels, height_in_pixels) = (dimensions[0], dimensions[1]);
        let width_in_points = width_in_pixels as f32 / pixels_per_point;
        let height_in_points = height_in_pixels as f32 / pixels_per_point;

        if let Some((view, _image)) = self.get_texture(mesh.texture_id) {

            //FIXME: annoying
            // PERFORMANCE_WARNING: "Texture state usage warning: The texture object (0) bound to texture image unit 0 does not have a defined base level and cannot be used for texture mapping." ??
            let screen_size = [width_in_points, height_in_points];

            let clip_min_x = pixels_per_point * clip_rect.min.x;
            let clip_min_y = pixels_per_point * clip_rect.min.y;
            let clip_max_x = pixels_per_point * clip_rect.max.x;
            let clip_max_y = pixels_per_point * clip_rect.max.y;

            // Make sure clip rect can fit within a `u32`:
            let clip_min_x = clip_min_x.clamp(0.0, width_in_pixels as f32);
            let clip_min_y = clip_min_y.clamp(0.0, height_in_pixels as f32);
            let clip_max_x = clip_max_x.clamp(clip_min_x, width_in_pixels as f32);
            let clip_max_y = clip_max_y.clamp(clip_min_y, height_in_pixels as f32);

            let clip_min_x = clip_min_x.round() as u32;
            let clip_min_y = clip_min_y.round() as u32;
            let clip_max_x = clip_max_x.round() as u32;
            let clip_max_y = clip_max_y.round() as u32;


            unsafe {
                device.set_viewport(0, &[Viewport {
                    x: 0.0,
                    y: 0.0,
                    w: width_in_pixels as f32,
                    h: height_in_pixels as f32,
                    n: 0.0,
                    f: 1.0,
                }]);
                device.set_scissor(0, &[Region {
                    x: clip_min_x as _,
                    y: (height_in_pixels - clip_max_y) as _,
                    w: (clip_max_x - clip_min_x) as _,
                    h: (clip_max_y - clip_min_y) as _,
                }]);


                device.bind_pipeline(self.pipeline);
                device.bind_color_blend_state(&ColorBlend{
                    attachments: vec![ColorBlendAttachment{
                        blend_enable: true,
                        color: BlendChannel {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::OneMinusSrcAlpha,
                            blend_op: BlendOp::Add
                        },
                        alpha: BlendChannel {
                            src_factor: BlendFactor::OneMinusDstAlpha,
                            dst_factor: BlendFactor::One,
                            blend_op: BlendOp::Add
                        }
                    }]
                });

                let sampler = self.sampler;

                device.bind_vertex_array(self.vertex_array);
                device.bind_vertex_buffers(self.vertex_array, 0, &[VertexBufferView {
                    buffer: vertex_buffer,
                    offset: 0,
                    stride: ((std::mem::size_of::<f32>() * 4) + (std::mem::size_of::<u8>() * 4)) as _,
                    input_rate: grr::InputRate::Vertex,
                }]);
                device.bind_uniform_constants(self.pipeline, 0, &[Constant::Vec2(screen_size)]);

                device.bind_samplers(0, &[sampler]);
                device.bind_image_views(0, &[*view]);

                device.bind_index_buffer(self.vertex_array, index_buffer);
                device.draw_indexed(grr::Primitive::Triangles, grr::IndexTy::U32, 0..(*indices as u32), 0..1, 0);

                device.clear_buffer(vertex_buffer, Format::R8_UNORM, BaseFormat::R, FormatLayout::U8, None);
                device.clear_buffer(index_buffer, Format::R8_UNORM, BaseFormat::R, FormatLayout::U8, None);
            }
        }
    }

    pub fn set_font_texture(
        &mut self,
        device: &grr::Device,
        tex_id: TextureId,
        delta: &FontImage,
    ) -> grr::Result<()> {

        let mut pixels = Vec::with_capacity(delta.height() * delta.width() * 4);
        for c in delta.srgba_pixels(1.0){
            pixels.push(c.r());
            pixels.push(c.g());
            pixels.push(c.b());
            pixels.push(c.a());
        }
        let width = delta.width();
        let height = delta.height();

        if let Some((_view, image)) = self.textures.get(&tex_id) {
            unsafe {
                device.copy_host_to_image(&pixels, *image, HostImageCopy{
                    host_layout: MemoryLayout {
                        base_format: BaseFormat::RGBA,
                        format_layout: FormatLayout::U8,
                        row_length: width as u32,
                        image_height: height as u32,
                        alignment: 4
                    },
                    image_subresource: SubresourceLayers { level: 0, layers: 0..1 },
                    image_offset: Offset {
                        x: 0 as _,
                        y: 0 as _,
                        z: 0
                    },
                    image_extent: Extent {
                        width: width as u32,
                        height: height as u32,
                        depth: 1
                    }
                });
            };
        } else {
            let img = unsafe { self.copy_host_to_tex(&device, &pixels, width as u32, height as u32)?};
            self.textures.insert(tex_id, img);
        }
        Ok(())
    }
    pub fn set_color_texture(&mut self, device : &grr::Device, id : TextureId, pixels : &[u8], width : u32, height : u32) -> grr::Result<()>{

        let tex = unsafe { self.copy_host_to_tex(device, &pixels, width, height) }? ;
        self.textures.insert(id, tex);
        Ok(())
    }

    pub unsafe fn free_texture(&mut self, tex_id: egui::TextureId, device: &grr::Device) {
        if let Some((view, img)) = self.textures.remove(&tex_id) {
            device.delete_image_view(view);
            device.delete_image(img);
        }
    }
    unsafe fn copy_host_to_tex(&mut self, device : &grr::Device, pixels : &[u8], width : u32, height : u32) -> grr::Result<(grr::ImageView, grr::Image)> {
        let (tex, view) = device.create_image_and_view(ImageType::D2 {
            width,
            height,
            layers: 1,
            samples: 1,
        }, grr::Format::R8G8B8A8_SRGB, 1)?;

        device.copy_host_to_image(pixels, tex, HostImageCopy {
            host_layout: grr::MemoryLayout {
                base_format: BaseFormat::RGBA,
                format_layout: FormatLayout::U8,
                row_length: width,
                image_height: height,
                alignment: 4,
            },
            image_subresource: grr::SubresourceLayers {
                level: 0,
                layers: 0..1,
            },
            image_offset: grr::Offset {
                x: 0,
                y: 0,
                z: 0,
            },
            image_extent: grr::Extent {
                width,
                height,
                depth: 1,
            },
        });
        Ok((view, tex))
    }

    fn get_texture(&self, texture_id: egui::TextureId) -> Option<&(grr::ImageView, grr::Image)> {
        self.textures.get(&texture_id)
    }
    pub unsafe fn free(&mut self, device : &grr::Device) {
        let mut keys = Vec::with_capacity(self.textures.len());
        for key in self.textures.keys(){
            keys.push(key.clone());
        }
        for i in keys{
            self.free_texture(i, device);
        }

        device.delete_buffer(self.vbo);
        device.delete_buffer(self.ibo);
        device.delete_sampler(self.sampler);
        device.delete_pipeline(self.pipeline);
        device.delete_vertex_array(self.vertex_array);
    }
}
