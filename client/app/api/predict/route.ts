import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { p1, p2, surface } = body
    
    if (!p1 || !p2 || !surface) {
      return NextResponse.json(
        { error: 'Missing required fields: p1, p2, surface' },
        { status: 400 }
      )
    }
    
    // Replace with your actual FastAPI server URL
    const apiUrl = process.env.FASTAPI_URL || 'http://127.0.0.1:8000'
    const response = await fetch(`${apiUrl}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ p1, p2, surface }),
    })
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error making prediction:', error)
    return NextResponse.json(
      { error: 'Failed to make prediction' },
      { status: 500 }
    )
  }
} 