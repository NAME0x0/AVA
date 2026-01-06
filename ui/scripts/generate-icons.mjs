/**
 * AVA Icon Generator
 * Generates all required icon sizes from the SVG source
 */

import sharp from 'sharp';
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const ICONS_DIR = join(__dirname, '..', 'src-tauri', 'icons');
const PUBLIC_DIR = join(__dirname, '..', 'public');
const SVG_SOURCE = join(ICONS_DIR, 'ava-icon.svg');

// Ensure directories exist
if (!existsSync(PUBLIC_DIR)) {
  mkdirSync(PUBLIC_DIR, { recursive: true });
}

// Icon sizes needed for Tauri
const TAURI_SIZES = [
  { name: '32x32.png', size: 32 },
  { name: '128x128.png', size: 128 },
  { name: '128x128@2x.png', size: 256 },
];

// Favicon sizes for web
const FAVICON_SIZES = [
  { name: 'favicon-16x16.png', size: 16 },
  { name: 'favicon-32x32.png', size: 32 },
  { name: 'apple-touch-icon.png', size: 180 },
  { name: 'android-chrome-192x192.png', size: 192 },
  { name: 'android-chrome-512x512.png', size: 512 },
];

async function generateIcons() {
  console.log('🎨 AVA Icon Generator');
  console.log('=====================\n');

  // Read SVG source
  const svgBuffer = readFileSync(SVG_SOURCE);
  console.log(`📂 Source: ${SVG_SOURCE}\n`);

  // Generate Tauri icons
  console.log('📱 Generating Tauri icons...');
  for (const { name, size } of TAURI_SIZES) {
    const outputPath = join(ICONS_DIR, name);
    await sharp(svgBuffer)
      .resize(size, size)
      .png()
      .toFile(outputPath);
    console.log(`   ✓ ${name} (${size}x${size})`);
  }

  // Generate ICO file (Windows) - multiple sizes embedded
  console.log('\n🪟 Generating Windows icon...');
  const icoSizes = [16, 32, 48, 64, 128, 256];
  const icoBuffers = await Promise.all(
    icoSizes.map(size =>
      sharp(svgBuffer)
        .resize(size, size)
        .png()
        .toBuffer()
    )
  );

  // Create ICO file manually (simplified - single 256x256 for now)
  const ico256 = await sharp(svgBuffer)
    .resize(256, 256)
    .png()
    .toFile(join(ICONS_DIR, 'icon.ico.png'));
  console.log('   ✓ icon.ico.png (convert manually or use png2ico)');

  // Generate ICNS placeholder info
  console.log('\n🍎 macOS icon...');
  await sharp(svgBuffer)
    .resize(512, 512)
    .png()
    .toFile(join(ICONS_DIR, 'icon.icns.png'));
  console.log('   ✓ icon.icns.png (convert with iconutil on macOS)');

  // Generate web favicons
  console.log('\n🌐 Generating web favicons...');
  for (const { name, size } of FAVICON_SIZES) {
    const outputPath = join(PUBLIC_DIR, name);
    await sharp(svgBuffer)
      .resize(size, size)
      .png()
      .toFile(outputPath);
    console.log(`   ✓ ${name} (${size}x${size})`);
  }

  // Copy SVG as favicon.svg
  writeFileSync(join(PUBLIC_DIR, 'favicon.svg'), svgBuffer);
  console.log('   ✓ favicon.svg');

  // Generate OG image (1200x630)
  console.log('\n🖼️ Generating social media image...');
  const ogWidth = 1200;
  const ogHeight = 630;
  const iconSize = 400;

  // Create OG image with centered icon on dark background
  const ogBackground = await sharp({
    create: {
      width: ogWidth,
      height: ogHeight,
      channels: 4,
      background: { r: 10, g: 10, b: 15, alpha: 1 }
    }
  }).png().toBuffer();

  const iconResized = await sharp(svgBuffer)
    .resize(iconSize, iconSize)
    .png()
    .toBuffer();

  await sharp(ogBackground)
    .composite([{
      input: iconResized,
      left: Math.floor((ogWidth - iconSize) / 2),
      top: Math.floor((ogHeight - iconSize) / 2 - 30),
    }])
    .png()
    .toFile(join(PUBLIC_DIR, 'og-image.png'));
  console.log('   ✓ og-image.png (1200x630)');

  // Create web manifest
  const manifest = {
    name: 'AVA Neural Interface',
    short_name: 'AVA',
    icons: [
      { src: '/android-chrome-192x192.png', sizes: '192x192', type: 'image/png' },
      { src: '/android-chrome-512x512.png', sizes: '512x512', type: 'image/png' }
    ],
    theme_color: '#00D4C8',
    background_color: '#0A0A0F',
    display: 'standalone'
  };
  writeFileSync(join(PUBLIC_DIR, 'site.webmanifest'), JSON.stringify(manifest, null, 2));
  console.log('   ✓ site.webmanifest');

  console.log('\n✨ Icon generation complete!\n');
  console.log('📝 Manual steps needed:');
  console.log('   1. Convert icon.ico.png to icon.ico using png2ico or online tool');
  console.log('   2. On macOS, convert icon.icns.png using iconutil');
  console.log('   3. Update ui/src/app/layout.tsx with favicon links\n');
}

generateIcons().catch(console.error);
