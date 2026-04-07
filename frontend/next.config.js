/** @type {import('next').NextConfig} */
const nextConfig = {
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
        source: '/tools',
        destination: 'http://localhost:7860/tools',
      },
      {
        source: '/ws/:path*',
        destination: 'http://localhost:7860/ws/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
