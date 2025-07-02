import { NextResponse } from 'next/server'

export async function GET() {
  try {
    // Replace with your actual FastAPI server URL
    const apiUrl = process.env.FASTAPI_URL || 'http://localhost:8000'
    const response = await fetch(`${apiUrl}/players`)
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error fetching players:', error)
    return NextResponse.json(
      { error: 'Failed to fetch players' },
      { status: 500 }
    )
  }
} 