/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export only in production builds (for Docker/HF Spaces).
  // In dev mode, rewrites work normally and proxy to the backend.
  ...(process.env.NODE_ENV === 'production' ? { output: 'export', distDir: 'dist' } : {}),
  turbopack: {
    root: '..',
  },
  async rewrites() {
    return [
      {
        source: '/env/:path*',
        destination: 'http://localhost:7860/env/:path*',
      },
      {
        source: '/episodes/:path*',
        destination: 'http://localhost:7860/episodes/:path*',
      },
      {
        source: '/episodes',
        destination: 'http://localhost:7860/episodes',
      },
      {
        source: '/tools',
        destination: 'http://localhost:7860/tools',
      },
      {
        source: '/ws/:path*',
        destination: 'http://localhost:7860/ws/:path*',
      },
      {
        source: '/marl/:path*',
        destination: 'http://localhost:7860/marl/:path*',
      },
      {
        source: '/curriculum/:path*',
        destination: 'http://localhost:7860/curriculum/:path*',
      },
      {
        source: '/selfplay/:path*',
        destination: 'http://localhost:7860/selfplay/:path*',
      },
      {
        source: '/demo/:path*',
        destination: 'http://localhost:7860/demo/:path*',
      },
      {
        source: '/health',
        destination: 'http://localhost:7860/health',
      },
      {
        source: '/mcp',
        destination: 'http://localhost:7860/mcp',
      },
    ];
  },
};

module.exports = nextConfig;
