# MatchPointAI Front-end Client

A modern, responsive Next.js web application for tennis match prediction. Built with TypeScript, Tailwind CSS, and Radix UI components, this frontend provides an intuitive interface for predicting tennis match outcomes using machine learning.

## 🎾 Features

- **Interactive Match Prediction**: Select players and surface type to get AI-powered predictions
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Real-time API Integration**: Seamless communication with the FastAPI backend
- **Modern UI/UX**: Beautiful, accessible interface with smooth animations
- **Player Database**: Comprehensive list of ATP players with statistics
- **Surface-specific Analysis**: Predictions tailored to different court surfaces (Hard, Clay, Grass, Carpet)

## 🏗️ Architecture

The frontend is built with modern web technologies:

- **Next.js 15**: React framework with App Router, using TypeScript
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Radix UI**: Accessible component primitives
- **Lucide React**: Beautiful icon library
- **Responsive Design**: Mobile-first approach with scroll optimization

## 📁 Project Structure

```
client/
├── app/                    # Next.js App Router
│   ├── api/               # API route handlers
│   │   ├── ping/          # Health check endpoint
│   │   ├── players/       # Player data endpoint
│   │   └── predict/       # Match prediction endpoint
│   ├── globals.css        # Global styles and Tailwind imports
│   ├── layout.tsx         # Root layout component
│   └── page.tsx           # Main application page
├── components/            # Reusable UI components
│   └── ui/               # Radix UI-based components
│       ├── button.tsx    # Button component
│       ├── card.tsx      # Card component
│       ├── input.tsx     # Input component
│       ├── label.tsx     # Label component
│       └── select.tsx    # Select dropdown component
├── lib/                  # Utility functions
│   └── utils.ts         # Helper functions
├── public/              # Static assets
├── package.json         # Dependencies and scripts
├── tailwind.config.ts   # Tailwind CSS configuration
└── tsconfig.json        # TypeScript configuration
```

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ 
- npm, yarn, pnpm, or bun
- Backend server running (see server README)

### Installation

1. **Navigate to the client directory:**
   ```bash
   cd client
   ```

2. **Install dependencies:**
   ```bash
   npm install
   # or
   yarn install
   # or
   pnpm install
   ```

3. **Set up environment variables:**
   Create a `.env.local` file in the client directory:
   ```env
   FASTAPI_URL=http://localhost:8000
   ```

4. **Start the development server:**
   ```bash
   npm run dev
   # or
   yarn dev
   # or
   pnpm dev
   ```

5. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

## 🎯 How to Use

### Making Predictions

1. **Select Players**: Choose two ATP players from the dropdown menus
2. **Choose Surface**: Select the court surface (Hard, Clay, Grass, or Carpet)
3. **Get Prediction**: Click "Predict Match" to receive AI-powered predictions
4. **View Results**: See the predicted winner and confidence percentage

### Features

- **Mobile-Optimized**: Smooth scrolling and touch-friendly interface
- **Real-time Updates**: Instant feedback during prediction process
- **Error Handling**: Graceful error messages for failed requests
- **Loading States**: Visual feedback during API calls

## 🔧 Development

### Available Scripts

```bash
npm run dev      # Start development server
npm run build    # Build for production
npm run start    # Start production server
npm run lint     # Run ESLint
```

### Key Components

#### Main Application (`app/page.tsx`)
- Hero section with feature highlights
- Interactive prediction form
- Results display with animations
- Mobile-responsive scroll behavior

#### API Routes (`app/api/`)
- **`/api/ping`**: Health check for backend connectivity
- **`/api/players`**: Fetch available players from database
- **`/api/predict`**: Submit prediction requests to ML model

#### UI Components (`components/ui/`)
- **Button**: Styled button with loading states
- **Card**: Container component for content sections
- **Select**: Dropdown for player and surface selection
- **Label**: Form labels with accessibility features

### Styling

The application uses Tailwind CSS with a custom color palette:
- **Primary**: `#3a4a3a` (Forest Green)
- **Secondary**: `#b8956a` (Warm Brown)
- **Accent**: `#e8dcc0` (Cream)
- **Background**: `#3a4a3a` (Dark Green)

### Mobile Optimization

- **Scroll Behavior**: Automatic scroll locking on desktop, free scrolling on mobile
- **Touch Targets**: Appropriately sized buttons and interactive elements
- **Responsive Layout**: Grid layouts that adapt to screen size
- **Performance**: Optimized animations and transitions

## 🌐 API Integration

### Backend Communication

The frontend communicates with the FastAPI backend through Next.js API routes:

```typescript
// Example: Making a prediction
const response = await fetch('/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ p1, p2, surface })
})
```

### Environment Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `FASTAPI_URL` | Backend server URL | `http://127.0.0.1:8000` |

## 🚀 Deployment

### Vercel (Recommended)

1. **Connect your repository to Vercel**
2. **Set environment variables** in Vercel dashboard
3. **Deploy automatically** on git push

### Other Platforms

The app can be deployed to any platform that supports Next.js:

```bash
npm run build
npm run start
```

### Environment Variables for Production

```env
FASTAPI_URL=https://your-backend-url.com
```

## 🎨 Customization

### Adding New Features

1. **New Pages**: Create new files in `app/` directory
2. **Components**: Add reusable components in `components/`
3. **API Routes**: Create new endpoints in `app/api/`
4. **Styling**: Modify `globals.css` or component styles

### Theming

Update the color scheme in `tailwind.config.ts`:

```typescript
theme: {
  extend: {
    colors: {
      primary: '#3a4a3a',
      secondary: '#b8956a',
      accent: '#e8dcc0',
    }
  }
}
```

## 🧪 Testing

### Manual Testing

1. **Player Selection**: Test dropdown functionality
2. **Surface Selection**: Verify all surface options work
3. **Prediction Flow**: Test complete prediction workflow
4. **Mobile Responsiveness**: Test on various screen sizes
5. **Error Handling**: Test with backend offline

### Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 📱 Mobile Features

- **Touch-Friendly Interface**: Large touch targets and smooth gestures
- **Responsive Design**: Adapts to all screen sizes
- **Scroll Optimization**: Natural scrolling behavior on mobile devices
- **Performance**: Optimized for mobile network conditions

## 🔒 Security

- **Input Validation**: Client-side validation for all form inputs
- **Error Handling**: Graceful error messages without exposing internals
- **CORS**: Properly configured for backend communication

## 📈 Performance

- **Next.js Optimization**: Automatic code splitting and optimization
- **Image Optimization**: Next.js Image component for optimal loading
- **Font Loading**: Optimized font loading with `next/font`
- **Bundle Analysis**: Use `npm run build` to analyze bundle size


