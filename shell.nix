{ pkgs ? import <nixpkgs> { }, lib ? pkgs.lib }:
pkgs.mkShell rec {
name = "rust-env";
buildInputs = with pkgs; [
cmake
gcc
mesa

# rustc
# cargo
clang
openssl
pkgconfig
git

xorg.libX11
xorg.libXcursor
xorg.libXrandr
xorg.libXi
# alsaLib
# freetype
# expat
vulkan-tools
vulkan-loader
vulkan-validation-layers
vulkan-tools-lunarg
vulkan-extension-layer

shaderc
shaderc.bin
shaderc.static
shaderc.dev
shaderc.lib
];
LD_LIBRARY_PATH = "${lib.makeLibraryPath buildInputs}";
VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
XDG_DATA_DIRS = builtins.getEnv "XDG_DATA_DIRS";
XDG_RUNTIME_DIR = builtins.getEnv "XDG_RUNTIME_DIR";
}
