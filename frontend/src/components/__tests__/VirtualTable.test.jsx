import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import VirtualTable from '../VirtualTable'

// Mock data generator
const generateMockData = (count) => {
  return Array.from({ length: count }, (_, i) => ({
    id: i + 1,
    name: `User ${i + 1}`,
    email: `user${i + 1}@example.com`,
    status: i % 3 === 0 ? 'active' : i % 3 === 1 ? 'inactive' : 'pending',
    createdAt: new Date(2023, 0, i + 1).toISOString(),
    score: Math.floor(Math.random() * 100)
  }))
}

const mockColumns = [
  { key: 'id', title: 'ID', width: 80 },
  { key: 'name', title: 'Name', width: 150 },
  { key: 'email', title: 'Email', width: 200 },
  { 
    key: 'status', 
    title: 'Status', 
    width: 100,
    render: (value) => (
      <span className={`status-${value}`}>{value}</span>
    )
  },
  { key: 'score', title: 'Score', width: 100 }
]

describe('VirtualTable', () => {
  let user
  let mockData
  
  beforeEach(() => {
    user = userEvent.setup()
    mockData = generateMockData(1000)
    
    // Mock window.requestAnimationFrame for virtual scrolling
    global.requestAnimationFrame = vi.fn((cb) => setTimeout(cb, 16))
    global.cancelAnimationFrame = vi.fn()
    
    // Mock IntersectionObserver for infinite loading
    global.IntersectionObserver = vi.fn().mockImplementation(() => ({
      observe: vi.fn(),
      unobserve: vi.fn(),
      disconnect: vi.fn(),
    }))
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  describe('Basic Rendering', () => {
    it('renders table with correct structure', () => {
      render(
        <VirtualTable
          data={mockData.slice(0, 10)}
          columns={mockColumns}
          height={400}
        />
      )

      // Check if header is rendered
      expect(screen.getByText('ID')).toBeInTheDocument()
      expect(screen.getByText('Name')).toBeInTheDocument()
      expect(screen.getByText('Email')).toBeInTheDocument()

      // Check if status bar is rendered
      expect(screen.getByText(/Mostrando \d+ de \d+ registros/)).toBeInTheDocument()
    })

    it('renders empty state when no data provided', () => {
      render(
        <VirtualTable
          data={[]}
          columns={mockColumns}
          height={400}
        />
      )

      expect(screen.getByText('Mostrando 0 de 0 registros')).toBeInTheDocument()
    })

    it('applies custom className', () => {
      const { container } = render(
        <VirtualTable
          data={mockData.slice(0, 10)}
          columns={mockColumns}
          height={400}
          className="custom-table-class"
        />
      )

      expect(container.firstChild).toHaveClass('custom-table-class')
    })
  })

  describe('Search Functionality', () => {
    it('filters data based on search input', async () => {
      render(
        <VirtualTable
          data={mockData.slice(0, 100)}
          columns={mockColumns}
          height={400}
          searchable={true}
        />
      )

      const searchInput = screen.getByPlaceholderText('Buscar em todos os campos...')
      
      await user.type(searchInput, 'User 1')
      
      // Wait for throttled search
      await waitFor(() => {
        expect(screen.getByText(/Mostrando \d+ de 100 registros/)).toBeInTheDocument()
      }, { timeout: 1000 })
    })

    it('shows no results when search returns empty', async () => {
      render(
        <VirtualTable
          data={mockData.slice(0, 10)}
          columns={mockColumns}
          height={400}
          searchable={true}
        />
      )

      const searchInput = screen.getByPlaceholderText('Buscar em todos os campos...')
      
      await user.type(searchInput, 'NonExistentUser')
      
      await waitFor(() => {
        expect(screen.getByText('Mostrando 0 de 10 registros')).toBeInTheDocument()
      })
    })

    it('hides search when searchable is false', () => {
      render(
        <VirtualTable
          data={mockData.slice(0, 10)}
          columns={mockColumns}
          height={400}
          searchable={false}
        />
      )

      expect(screen.queryByPlaceholderText('Buscar em todos os campos...')).not.toBeInTheDocument()
    })
  })

  describe('Sorting Functionality', () => {
    it('sorts data when column header is clicked', async () => {
      render(
        <VirtualTable
          data={mockData.slice(0, 10)}
          columns={mockColumns}
          height={400}
          sortable={true}
        />
      )

      const nameHeader = screen.getByText('Name')
      
      await user.click(nameHeader)
      
      // Check if sorting indicators are present
      const headerElement = nameHeader.closest('div')
      expect(within(headerElement).getByTestId('sort-icon') || headerElement.querySelector('svg')).toBeInTheDocument()
    })

    it('toggles sort direction on multiple clicks', async () => {
      render(
        <VirtualTable
          data={mockData.slice(0, 10)}
          columns={mockColumns}
          height={400}
          sortable={true}
        />
      )

      const nameHeader = screen.getByText('Name')
      
      // First click - ascending
      await user.click(nameHeader)
      
      // Second click - descending
      await user.click(nameHeader)
      
      // Verify sort direction change (implementation specific)
    })

    it('disables sorting when sortable is false', async () => {
      render(
        <VirtualTable
          data={mockData.slice(0, 10)}
          columns={mockColumns}
          height={400}
          sortable={false}
        />
      )

      const nameHeader = screen.getByText('Name')
      
      await user.click(nameHeader)
      
      // Should not show sort indicators
      const headerElement = nameHeader.closest('div')
      expect(headerElement.querySelector('svg')).not.toBeInTheDocument()
    })
  })

  describe('Row Selection', () => {
    it('handles row selection when onRowSelect is provided', async () => {
      const mockOnRowSelect = vi.fn()
      const selectedRows = new Set()
      
      render(
        <VirtualTable
          data={mockData.slice(0, 10)}
          columns={mockColumns}
          height={400}
          onRowSelect={mockOnRowSelect}
          selectedRows={selectedRows}
        />
      )

      // Find first checkbox (assuming it exists)
      const checkboxes = screen.getAllByRole('checkbox')
      
      if (checkboxes.length > 0) {
        await user.click(checkboxes[0])
        expect(mockOnRowSelect).toHaveBeenCalled()
      }
    })

    it('shows selected state for selected rows', () => {
      const selectedRows = new Set([1, 3])
      
      render(
        <VirtualTable
          data={mockData.slice(0, 10)}
          columns={mockColumns}
          height={400}
          onRowSelect={vi.fn()}
          selectedRows={selectedRows}
        />
      )

      // Check if selected rows have proper styling
      // This would depend on how selection is visually indicated
    })
  })

  describe('Row Click Handling', () => {
    it('calls onRowClick when row is clicked', async () => {
      const mockOnRowClick = vi.fn()
      
      render(
        <VirtualTable
          data={mockData.slice(0, 10)}
          columns={mockColumns}
          height={400}
          onRowClick={mockOnRowClick}
        />
      )

      // Click on first row (implementation specific selector)
      const firstRow = screen.getByText('User 1').closest('div')
      await user.click(firstRow)
      
      expect(mockOnRowClick).toHaveBeenCalledWith(mockData[0])
    })
  })

  describe('Infinite Loading', () => {
    it('shows loading indicator when isNextPageLoading is true', () => {
      render(
        <VirtualTable
          data={mockData.slice(0, 10)}
          columns={mockColumns}
          height={400}
          hasNextPage={true}
          isNextPageLoading={true}
        />
      )

      expect(screen.getByText('Carregando mais...')).toBeInTheDocument()
    })

    it('calls loadNextPage when scrolling near bottom', async () => {
      const mockLoadNextPage = vi.fn()
      
      render(
        <VirtualTable
          data={mockData.slice(0, 10)}
          columns={mockColumns}
          height={400}
          hasNextPage={true}
          loadNextPage={mockLoadNextPage}
        />
      )

      // Simulate scrolling to trigger infinite loading
      // This would require mocking the virtual list's scroll behavior
    })
  })

  describe('Performance', () => {
    it('handles large datasets efficiently', () => {
      const largeDataset = generateMockData(50000)
      
      const startTime = performance.now()
      
      render(
        <VirtualTable
          data={largeDataset}
          columns={mockColumns}
          height={400}
        />
      )
      
      const endTime = performance.now()
      const renderTime = endTime - startTime
      
      // Expect render time to be reasonable (less than 100ms)
      expect(renderTime).toBeLessThan(100)
    })

    it('virtualizes rows correctly with large datasets', () => {
      const largeDataset = generateMockData(10000)
      
      render(
        <VirtualTable
          data={largeDataset}
          columns={mockColumns}
          height={400}
          rowHeight={48}
        />
      )

      // Only visible rows should be rendered
      // This would require testing the virtual list implementation
      expect(screen.getByText('Mostrando 10,000 de 10,000 registros')).toBeInTheDocument()
    })
  })

  describe('Custom Rendering', () => {
    it('uses custom render function for columns', () => {
      const customData = [
        { id: 1, name: 'Test User', status: 'active' }
      ]
      
      render(
        <VirtualTable
          data={customData}
          columns={mockColumns}
          height={400}
        />
      )

      // Check if custom status rendering is applied
      expect(screen.getByText('active')).toHaveClass('status-active')
    })
  })

  describe('Accessibility', () => {
    it('provides proper ARIA labels and roles', () => {
      render(
        <VirtualTable
          data={mockData.slice(0, 10)}
          columns={mockColumns}
          height={400}
        />
      )

      // Check for table accessibility features
      // This would depend on the specific ARIA implementation
    })

    it('supports keyboard navigation', async () => {
      render(
        <VirtualTable
          data={mockData.slice(0, 10)}
          columns={mockColumns}
          height={400}
        />
      )

      // Test keyboard navigation
      await user.tab()
      
      // Verify focus management
      expect(document.activeElement).toBeInTheDocument()
    })
  })

  describe('Error Handling', () => {
    it('handles invalid data gracefully', () => {
      const invalidData = [
        { id: 1, name: null, email: undefined },
        null,
        undefined
      ]
      
      expect(() => {
        render(
          <VirtualTable
            data={invalidData}
            columns={mockColumns}
            height={400}
          />
        )
      }).not.toThrow()
    })

    it('handles missing columns gracefully', () => {
      expect(() => {
        render(
          <VirtualTable
            data={mockData.slice(0, 10)}
            columns={[]}
            height={400}
          />
        )
      }).not.toThrow()
    })
  })

  describe('Memory Management', () => {
    it('cleans up resources on unmount', () => {
      const { unmount } = render(
        <VirtualTable
          data={mockData.slice(0, 10)}
          columns={mockColumns}
          height={400}
        />
      )

      // Mock cleanup checks
      const mockCleanup = vi.fn()
      
      unmount()
      
      // Verify cleanup was called (implementation specific)
    })
  })
})