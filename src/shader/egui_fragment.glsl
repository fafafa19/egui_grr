#version 460 core

layout (binding = 0) uniform sampler2D u_sampler;

layout (location = 0) in vec4 a_rgba;
layout (location = 1) in vec2 a_tc;

out vec4 frag_color;

void main() {
    // The texture is set up with `SRGB8_ALPHA8`, so no need to decode here!
    vec4 texture_rgba = texture(u_sampler, a_tc);
    /// Multiply vertex color with texture color (in linear space).
    frag_color = a_rgba * texture_rgba;
}