import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.js'],
    globals: true,
    css: true,
    coverage: {
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/test/',
        '**/*.d.ts',
        '**/*.config.js',
        '**/*.test.{js,jsx,ts,tsx}',
        '**/index.js'
      ],
      thresholds: {
        global: {
          branches: 80,
          functions: 80,
          lines: 80,
          statements: 80
        }
      }
    },
    // Performance testing configuration
    benchmark: {
      include: ['**/*.bench.{js,jsx,ts,tsx}'],
      exclude: ['node_modules/**/*']
    },
    // Mock service workers and browser APIs
    deps: {
      inline: ['@testing-library/user-event']
    },
    // Test timeout for performance tests
    testTimeout: 30000,
    hookTimeout: 30000
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  define: {
    // Mock environment variables for testing
    __PERFORMANCE_MONITORING__: false,
    __BUILD_TIME__: JSON.stringify('test-build')
  }
})