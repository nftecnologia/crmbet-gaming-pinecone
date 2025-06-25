import React, { memo, useMemo, useCallback, useRef, useEffect, useState } from 'react'
import { FixedSizeList as List } from 'react-window'
import InfiniteLoader from 'react-window-infinite-loader'
import AutoSizer from 'react-virtualized-auto-sizer'
import throttle from 'lodash.throttle'
import { ChevronDown, ChevronUp, Search, Filter } from 'lucide-react'

// Ultra-optimized Row Component with React.memo
const VirtualRow = memo(({ index, style, data }) => {
  const { items, columns, onRowClick, selectedRows, onRowSelect } = data
  const item = items[index]
  
  if (!item) {
    return (
      <div style={style} className="flex items-center justify-center">
        <div className="animate-pulse bg-gray-200 h-8 w-full rounded"></div>
      </div>
    )
  }

  const isSelected = selectedRows?.has(item.id)
  
  return (
    <div
      style={style}
      className={`flex items-center border-b border-gray-100 hover:bg-gray-50 cursor-pointer transition-colors duration-75 ${
        isSelected ? 'bg-blue-50 border-blue-200' : ''
      }`}
      onClick={() => onRowClick?.(item)}
    >
      {onRowSelect && (
        <div className="w-12 flex justify-center">
          <input
            type="checkbox"
            checked={isSelected}
            onChange={(e) => {
              e.stopPropagation()
              onRowSelect(item.id, e.target.checked)
            }}
            className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
          />
        </div>
      )}
      {columns.map((column) => (
        <div
          key={column.key}
          className={`px-4 py-3 text-sm ${column.className || ''}`}
          style={{ width: column.width, minWidth: column.minWidth }}
        >
          {column.render ? column.render(item[column.key], item) : item[column.key]}
        </div>
      ))}
    </div>
  )
})

VirtualRow.displayName = 'VirtualRow'

// Ultra-performance Virtual Table Component
const VirtualTable = ({
  data = [],
  columns = [],
  height = 600,
  rowHeight = 48,
  hasNextPage = false,
  isNextPageLoading = false,
  loadNextPage = () => {},
  onRowClick,
  onRowSelect,
  selectedRows = new Set(),
  searchable = true,
  filterable = true,
  sortable = true,
  className = ''
}) => {
  const [searchTerm, setSearchTerm] = useState('')
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' })
  const [filters, setFilters] = useState({})
  const listRef = useRef()
  
  // Throttled search to prevent excessive filtering
  const throttledSearch = useMemo(
    () => throttle((term) => {
      setSearchTerm(term)
    }, 300),
    []
  )

  // Memoized filtered and sorted data
  const processedData = useMemo(() => {
    let filtered = data

    // Apply search filter
    if (searchTerm) {
      const searchLower = searchTerm.toLowerCase()
      filtered = filtered.filter(item =>
        columns.some(column => {
          const value = item[column.key]
          return value && value.toString().toLowerCase().includes(searchLower)
        })
      )
    }

    // Apply column filters
    Object.entries(filters).forEach(([key, value]) => {
      if (value) {
        filtered = filtered.filter(item => {
          const itemValue = item[key]
          if (typeof value === 'string') {
            return itemValue && itemValue.toString().toLowerCase().includes(value.toLowerCase())
          }
          return itemValue === value
        })
      }
    })

    // Apply sorting
    if (sortConfig.key) {
      filtered.sort((a, b) => {
        const aVal = a[sortConfig.key]
        const bVal = b[sortConfig.key]
        
        if (aVal < bVal) return sortConfig.direction === 'asc' ? -1 : 1
        if (aVal > bVal) return sortConfig.direction === 'asc' ? 1 : -1
        return 0
      })
    }

    return filtered
  }, [data, searchTerm, sortConfig, filters, columns])

  // Infinite loading logic
  const itemCount = hasNextPage ? processedData.length + 1 : processedData.length
  const isItemLoaded = useCallback((index) => {
    return !!processedData[index]
  }, [processedData])

  // Handle sorting
  const handleSort = useCallback((columnKey) => {
    if (!sortable) return
    
    setSortConfig(prev => ({
      key: columnKey,
      direction: prev.key === columnKey && prev.direction === 'asc' ? 'desc' : 'asc'
    }))
  }, [sortable])

  // Handle column filtering
  const handleFilter = useCallback((columnKey, value) => {
    setFilters(prev => ({
      ...prev,
      [columnKey]: value
    }))
  }, [])

  // Row data for virtualization
  const rowData = useMemo(() => ({
    items: processedData,
    columns,
    onRowClick,
    onRowSelect,
    selectedRows
  }), [processedData, columns, onRowClick, onRowSelect, selectedRows])

  // Header component
  const TableHeader = memo(() => (
    <div className="flex items-center bg-gray-50 border-b-2 border-gray-200 font-medium text-gray-700">
      {onRowSelect && <div className="w-12"></div>}
      {columns.map((column) => (
        <div
          key={column.key}
          className={`px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100 transition-colors ${
            column.className || ''
          }`}
          style={{ width: column.width, minWidth: column.minWidth }}
          onClick={() => column.sortable !== false && handleSort(column.key)}
        >
          <div className="flex items-center justify-between">
            <span>{column.title}</span>
            {sortable && column.sortable !== false && (
              <div className="flex flex-col">
                <ChevronUp 
                  className={`w-3 h-3 ${
                    sortConfig.key === column.key && sortConfig.direction === 'asc' 
                      ? 'text-blue-600' 
                      : 'text-gray-300'
                  }`} 
                />
                <ChevronDown 
                  className={`w-3 h-3 -mt-1 ${
                    sortConfig.key === column.key && sortConfig.direction === 'desc' 
                      ? 'text-blue-600' 
                      : 'text-gray-300'
                  }`} 
                />
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  ))

  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 ${className}`}>
      {/* Search and Filter Bar */}
      {(searchable || filterable) && (
        <div className="p-4 border-b border-gray-200 bg-gray-50">
          <div className="flex items-center space-x-4">
            {searchable && (
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <input
                  type="text"
                  placeholder="Buscar em todos os campos..."
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  onChange={(e) => throttledSearch(e.target.value)}
                />
              </div>
            )}
            {filterable && (
              <button className="flex items-center px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50 transition-colors">
                <Filter className="w-4 h-4 mr-2" />
                Filtros
              </button>
            )}
          </div>
        </div>
      )}

      {/* Table Container */}
      <div style={{ height }}>
        <TableHeader />
        <div style={{ height: height - 48 }}>
          <AutoSizer>
            {({ height: autoHeight, width }) => (
              <InfiniteLoader
                isItemLoaded={isItemLoaded}
                itemCount={itemCount}
                loadMoreItems={loadNextPage}
                threshold={10}
              >
                {({ onItemsRendered, ref }) => (
                  <List
                    ref={(list) => {
                      ref(list)
                      listRef.current = list
                    }}
                    height={autoHeight}
                    width={width}
                    itemCount={itemCount}
                    itemSize={rowHeight}
                    itemData={rowData}
                    onItemsRendered={onItemsRendered}
                    overscanCount={5}
                  >
                    {VirtualRow}
                  </List>
                )}
              </InfiniteLoader>
            )}
          </AutoSizer>
        </div>
      </div>

      {/* Status Bar */}
      <div className="px-4 py-2 border-t border-gray-200 bg-gray-50 text-sm text-gray-600">
        Mostrando {processedData.length.toLocaleString()} de {data.length.toLocaleString()} registros
        {isNextPageLoading && (
          <span className="ml-2 text-blue-600">Carregando mais...</span>
        )}
      </div>
    </div>
  )
}

export default memo(VirtualTable)