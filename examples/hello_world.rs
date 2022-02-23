use grr::{BlendChannel, BlendFactor, BlendOp, ClearAttachment, ColorBlend, ColorBlendAttachment, Framebuffer, Region, Viewport};
use raw_gl_context::{GlConfig, GlContext, Profile};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use egui::CtxRef;
use epaint::TextureId;
use egui_grr::{Painter, PainterSettings};


fn main() -> anyhow::Result<()> {
    unsafe {
        let event_loop = EventLoop::new();

        let window = WindowBuilder::new()
            .with_title("egui")
            .with_inner_size(LogicalSize::new(1024.0, 768.0))
            .build(&event_loop)?;

        let context = GlContext::create(
            &window,
            GlConfig {
                version: (4, 6),
                profile: Profile::Core,
                red_bits: 8,
                blue_bits: 8,
                green_bits: 8,
                alpha_bits: 0,
                depth_bits: 0,
                stencil_bits: 0,
                samples: None,
                srgb: true,
                double_buffer: true,
                vsync: true,
            },
        )
            .unwrap();

        context.make_current();

        let mut ctx = CtxRef::default();

        let mut winit = egui_winit::State::new(&window);

        let grr = grr::Device::new(
            |symbol| context.get_proc_address(symbol) as *const _,
            grr::Debug::Enable {
                callback: |report, _, _, _, msg| {
                    println!("{:?}: {:?}", report, msg);
                },
                flags: grr::DebugReport::FULL,
            },
        );

        let pixels : Vec<u8> = image::open("examples/grr.png").unwrap().to_rgba8().to_vec();
        let mut painter = Painter::new(&grr, PainterSettings{
            ibo_size: 10000,
            vbo_size: 10000,
            max_texture_side: 4096
        })?;

        let mut first_paint = true;


        let mut text = String::with_capacity(128);

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => *control_flow = ControlFlow::Exit,
                Event::WindowEvent {
                    event,
                    ..
                } => {
                    winit.on_event(&ctx, &event);

                }
                Event::LoopDestroyed => {
                    painter.free(&grr);
                }
                Event::RedrawRequested(_) => {
                    {

                        grr.set_viewport(0, &[Viewport{
                            x: 0.0,
                            y: 0.0,
                            w: window.inner_size().width as _,
                            h: window.inner_size().height as _,
                            n: 0.0,
                            f: 1.0
                        }]);

                        grr.set_scissor(0, &[Region {
                            x: 0,
                            y: 0,
                            w: window.inner_size().width as _,
                            h: window.inner_size().height as _,
                        }]);
                        grr.clear_attachment(Framebuffer::DEFAULT, ClearAttachment::ColorFloat(0, [1.0, 1.0, 1.0, 1.0]));
                        grr.clear_attachment(Framebuffer::DEFAULT, ClearAttachment::DepthStencil(1.0, 0));

                        let input = winit.take_egui_input(&window);

                        let (output, shapes) = ctx.run(input, |ui|{
                            egui::Area::new("hi").show(&ui, |ui| {
                                ui.heading("hello world!");
                                if ui.button("quit").clicked() {
                                    *control_flow = ControlFlow::Exit;
                                }
                                ui.image(TextureId::User(0), (454.0, 302.0));
                                ui.code_editor(&mut text);
                            });
                        });
                        if first_paint{
                            painter.set_font_texture(&grr, TextureId::Egui, &ctx.font_image()).unwrap();
                            painter.set_user_texture(&grr, 0, &pixels, 454, 302).unwrap();
                            first_paint = !first_paint;
                        }

                        winit.handle_output(&window, &ctx, output);


                        let meshes = ctx.tessellate(shapes);
                        // draw things behind egui here
                        painter.paint(&grr, ctx.pixels_per_point(), window.inner_size().into(), meshes);

                        // draw things on top of egui here
                        context.swap_buffers();


                    }
                }
                Event::MainEventsCleared => {
                    window.request_redraw();
                }
                _ => (),
            }
        });
    }
}