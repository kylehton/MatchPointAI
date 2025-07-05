"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ChevronDown, TrendingUp, Users, Target, Loader2 } from "lucide-react"

interface PredictionResult {
  winner: string
  win_probability: number
}

// Helper function to detect mobile devices
const isMobileDevice = () => {
  if (typeof window === 'undefined') return false;
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) ||
         window.innerWidth <= 768;
}

export default function TennisPredictionApp() {
  const [player1, setPlayer1] = useState("")
  const [player2, setPlayer2] = useState("")
  const [surface, setSurface] = useState("")
  const [scrollLocked, setScrollLocked] = useState(true)
  const [predictionLoading, setPredictionLoading] = useState(false)
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [predictionError, setPredictionError] = useState("")
  const [players, setPlayers] = useState<string[]>([])
  const subpageRef = useRef<HTMLDivElement>(null);

  // Lock scroll on component mount (desktop only)
  useEffect(() => {
    const isMobile = isMobileDevice();
    
    if (scrollLocked && !isMobile) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = 'unset'
    }

    // Cleanup function to restore scroll when component unmounts
    return () => {
      document.body.style.overflow = 'unset'
    }
  }, [scrollLocked])

  useEffect(() => {
    const fetchPlayers = async () => {
      try {
        const response = await fetch("/api/players");
        if (!response.ok) throw new Error("Failed to fetch players");
        const data = await response.json();
        setPlayers(data.players || []);
      } catch (err) {
        console.log(err)
        throw new Error('Could not load players.');
      }
    };
    fetchPlayers();
  }, []);

  useEffect(() => {
    // Fire and forget ping to warm up the backend
    fetch("/api/ping").catch(() => {});
  }, []);

  const scrollToInput = () => {
    const isMobile = isMobileDevice();
    
    if (!isMobile) {
      setScrollLocked(false); // Unlock scroll to allow smooth scroll
    }
    
    const inputSection = document.getElementById("input-section");
    if (inputSection) {
      inputSection.scrollIntoView({ behavior: "smooth" });
      // After the scroll animation (e.g., 800ms), lock scroll again (desktop only)
      if (!isMobile) {
        setTimeout(() => {
          setScrollLocked(true);
        }, 800); // Adjust this duration to match your scroll speed
      }
    }
  }

  const handlePredict = async () => {
    if (!player1 || !player2 || !surface) return
    
    try {
      setPredictionLoading(true)
      setPredictionError("")
      setPredictionResult(null)
      
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          p1: player1,
          p2: player2,
          surface: surface
        }),
      })
      
      if (!response.ok) {
        throw new Error('Failed to make prediction')
      }
      
      const data = await response.json()
      console.log("Response from model:", data)
      if (data.error) {
        throw new Error(data.error)
      }
      
      setPredictionResult(data)
      const isMobile = isMobileDevice();
      
      setTimeout(() => {
        subpageRef.current?.scrollIntoView({ behavior: "smooth" });
        if (!isMobile) {
          setTimeout(() => {
            setScrollLocked(true);
          }, 800); // lock scroll after scroll animation (desktop only)
        }
      }, 100);
    } catch (err) {
      setPredictionError(err instanceof Error ? err.message : 'Failed to make prediction')
    } finally {
      setPredictionLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-[#3a4a3a]">
      {/* Hero Section */}
      <section className="min-h-screen flex flex-col items-center justify-center px-4 text-center">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="bg-[#e8dcc0] rounded-xl p-8 md:p-12 shadow-2xl">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="bg-[#4a5a4a] rounded-2xl p-6 text-white">
                <TrendingUp className="w-8 h-8 mb-4 text-[#e8dcc0]" />
                <h3 className="text-lg font-semibold mb-2">Advanced Analytics</h3>
                <p className="text-sm opacity-90">
                  Leveraging machine learning to analyze player statistics and match history
                </p>
              </div>
              <div className="bg-[#b8956a] rounded-2xl p-6 text-white">
                <Users className="w-8 h-8 mb-4 text-white" />
                <h3 className="text-lg font-semibold mb-2">Player Insights</h3>
                <p className="text-sm opacity-90">
                  Deep analysis of playing styles, strengths, and performance patterns
                </p>
              </div>
              <div className="bg-[#4a5a4a] rounded-2xl p-6 text-white">
                <Target className="w-8 h-8 mb-4 text-[#e8dcc0]" />
                <h3 className="text-lg font-semibold mb-2">Match Prediction</h3>
                <p className="text-sm opacity-90">
                  Accurate predictions based on surface type and head-to-head records
                </p>
              </div>
            </div>
            <div className="text-center">
              <h2 className="text-3xl md:text-4xl font-bold text-[#3a4a3a] mb-4">Predict Tennis Match Outcomes</h2>
              <p className="text-lg text-[#3a4a3a] opacity-80 mb-8 max-w-2xl mx-auto">
                Our advanced machine learning model analyzes player statistics, playing styles, and surface preferences
                to provide accurate match predictions.
              </p>
              <Button
                onClick={scrollToInput}
                className="bg-[#3a4a3a] hover:bg-[#2a3a2a] text-white px-8 py-6 text-lg rounded-full transition-all duration-300 hover:scale-105"
              >
                Start Prediction
                <ChevronDown className="ml-2 w-5 h-5 animate-bounce" />
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Input Section */}
      <section id="input-section" className="min-h-screen py-16 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">Match Setup</h2>
            <p className="text-xl text-white opacity-80">Enter player details and match conditions</p>
          </div>
          <div className="bg-[#e8dcc0] rounded-xl p-6 md:p-8 shadow-2xl">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Player 1 Section */}
              <Card className="bg-[#4a5a4a] border-none shadow-lg">
                <CardHeader className="text-center">
                  <CardTitle className="text-2xl text-white">Player 1</CardTitle>
                  <CardDescription className="text-[#e8dcc0]">Select first player</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="player1-select" className="text-white text-sm font-medium">
                      Player Name
                    </Label>
                    <Select value={player1} onValueChange={setPlayer1}>
                      <SelectTrigger className="mt-2 bg-white/10 border-white/20 text-white">
                        <SelectValue className='placeholder:text-white' placeholder="Choose player 1" />
                      </SelectTrigger>
                      <SelectContent className="bg-white text-black max-h-60">
                        {players.map((player) => (
                          <SelectItem key={player} value={player} className="hover:bg-gray-100">
                            {player}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  
                </CardContent>
              </Card>

              {/* Surface Type Section */}
              <Card className="bg-[#b8956a] border-none shadow-lg">
                <CardHeader className="text-center">
                  <CardTitle className="text-2xl text-white">Surface</CardTitle>
                  <CardDescription className="text-white/80">Select playing surface</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div>
                    <Label htmlFor="surface-type" className="text-white text-sm font-medium">
                      Court Surface
                    </Label>
                    <Select value={surface} onValueChange={setSurface}>
                      <SelectTrigger className="mt-2 bg-white/10 border-white/20 text-white">
                        <SelectValue className='text-white' placeholder="Choose surface" />
                      </SelectTrigger>
                      <SelectContent className="bg-white text-black">
                        <SelectItem value="Hard" className="hover:bg-gray-100">Hard</SelectItem>
                        <SelectItem value="Clay" className="hover:bg-gray-100">Clay</SelectItem>
                        <SelectItem value="Grass" className="hover:bg-gray-100">Grass</SelectItem>
                        <SelectItem value="Carpet" className="hover:bg-gray-100">Carpet</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  
                  <div className="pt-4">
                    <Button
                      onClick={handlePredict}
                      className="w-full bg-[#3a4a3a] hover:bg-[#2a3a2a] text-white py-3 text-lg rounded-xl transition-all duration-300 hover:scale-105"
                      disabled={!player1 || !player2 || !surface || predictionLoading}
                    >
                      {predictionLoading ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Predicting...
                        </>
                      ) : (
                        'Predict Match'
                      )}
                    </Button>
                  </div>
                  
                  {/* Prediction Results */}
                  {predictionError && (
                    <div className="mt-4 p-4 bg-red-100 border border-red-400 text-red-700 rounded-lg">
                      <p className="font-medium">Prediction Error:</p>
                      <p>{predictionError}</p>
                    </div>
                  )}
                  
                  {predictionResult && (
                    <div className="mt-4 p-4 bg-green-100 border border-green-400 text-green-700 rounded-lg">
                      <h4 className="font-bold text-lg mb-2">Prediction Result</h4>
                      <div className="space-y-2">
                        <p><strong>Winner:</strong> {predictionResult.winner}</p>
                        <p><strong>Confidence:</strong> {(predictionResult.win_probability).toFixed(2)}%</p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Player 2 Section */}
              <Card className="bg-[#4a5a4a] border-none shadow-lg">
                <CardHeader className="text-center">
                  <CardTitle className="text-2xl text-white">Player 2</CardTitle>
                  <CardDescription className="text-[#e8dcc0]">Select second player</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="player2-select" className="text-white text-sm font-medium">
                      Player Name
                    </Label>
                    <Select value={player2} onValueChange={setPlayer2}>
                      <SelectTrigger className="mt-2 bg-white/10 border-white/20 text-white">
                        <SelectValue className='text-white' placeholder="Choose player 2" />
                      </SelectTrigger>
                      <SelectContent className="bg-white text-black max-h-60">
                        {players.map((player) => (
                          <SelectItem
                            key={player}
                            value={player}
                            className="hover:bg-gray-100"
                            disabled={player === player1}
                          >
                            {player}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}