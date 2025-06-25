import '@testing-library/jest-dom'
import { beforeAll, afterEach, afterAll, vi } from 'vitest'
import { cleanup } from '@testing-library/react'

// Mock browser APIs
const mockCanvas = {
  getContext: vi.fn(() => ({
    fillRect: vi.fn(),
    clearRect: vi.fn(),
    getImageData: vi.fn(() => ({ data: new Array(4) })),
    putImageData: vi.fn(),
    createImageData: vi.fn(() => ({ data: new Array(4) })),
    setTransform: vi.fn(),
    drawImage: vi.fn(),
    save: vi.fn(),
    fillText: vi.fn(),
    restore: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    closePath: vi.fn(),
    stroke: vi.fn(),
    fill: vi.fn(),
    arc: vi.fn(),
    scale: vi.fn(),
    rotate: vi.fn(),
    translate: vi.fn(),
  })),
  toDataURL: vi.fn(() => 'data:image/png;base64,'),
  toBlob: vi.fn(),
}

global.HTMLCanvasElement.prototype.getContext = mockCanvas.getContext
global.HTMLCanvasElement.prototype.toDataURL = mockCanvas.toDataURL

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock IntersectionObserver
global.IntersectionObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))

// Mock WebSocket
global.WebSocket = vi.fn().mockImplementation(() => ({
  close: vi.fn(),
  send: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  CONNECTING: 0,
  OPEN: 1,
  CLOSING: 2,
  CLOSED: 3,
  readyState: 1,
}))

// Mock IndexedDB
const mockIDBDatabase = {
  transaction: vi.fn(() => ({
    objectStore: vi.fn(() => ({
      get: vi.fn(),
      put: vi.fn(),
      delete: vi.fn(),
      clear: vi.fn(),
      openCursor: vi.fn(),
      createIndex: vi.fn(),
      index: vi.fn(() => ({
        getAll: vi.fn(),
        count: vi.fn(),
      })),
    })),
  })),
  close: vi.fn(),
}

global.indexedDB = {
  open: vi.fn(() => ({
    onsuccess: null,
    onerror: null,
    onupgradeneeded: null,
    result: mockIDBDatabase,
  })),
  deleteDatabase: vi.fn(),
}

// Mock Worker
global.Worker = vi.fn().mockImplementation(() => ({
  postMessage: vi.fn(),
  terminate: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
}))

// Mock fetch
global.fetch = vi.fn()

// Mock URL.createObjectURL
global.URL.createObjectURL = vi.fn(() => 'blob:mock-url')
global.URL.revokeObjectURL = vi.fn()

// Mock navigator
Object.defineProperty(navigator, 'onLine', {
  writable: true,
  value: true,
})

Object.defineProperty(navigator, 'storage', {
  value: {
    estimate: vi.fn(() => Promise.resolve({ usage: 1024 * 1024 })),
  },
})

// Mock performance
global.performance.mark = vi.fn()
global.performance.measure = vi.fn()
global.performance.getEntriesByName = vi.fn(() => [])
global.performance.getEntriesByType = vi.fn(() => [])

// Mock requestAnimationFrame
global.requestAnimationFrame = vi.fn((callback) => {
  setTimeout(callback, 16)
  return 1
})

global.cancelAnimationFrame = vi.fn()

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(), // deprecated
    removeListener: vi.fn(), // deprecated
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
})

// Setup and teardown
beforeAll(() => {
  // Performance mark for test suite
  performance.mark('test-suite-start')
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

afterAll(() => {
  performance.mark('test-suite-end')
  performance.measure('test-suite-duration', 'test-suite-start', 'test-suite-end')
})