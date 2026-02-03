import { useState } from 'react'
import axios from 'axios'
import './App.css'
import ReactMarkdown from 'react-markdown';

const apiClient = axios.create({
  baseURL: 'http://127.0.0.1:5000',
})

// Define props type
interface ChatResponseProps {
  llmResponse: string;
}

function ChatResponse({ llmResponse }: ChatResponseProps) {
  return (
    <div className="response-container" style={{
      textAlign: 'left',  // Fix center alignment
      maxWidth: '800px',  // Constrain width for readability
      margin: '0 auto'    // Center the container itself (optional)
    }}>
      <ReactMarkdown
        components={{
          // Headings
          h1: ({...props}) => (
            <h1 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '16px' }} {...props} />
          ),
          h2: ({...props}) => (
            <h2 style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '12px' }} {...props} />
          ),
          
          // Paragraphs
          p: ({...props}) => (
            <p style={{ marginBottom: '12px', lineHeight: '1.6' }} {...props} />
          ),
          
          // Lists - THIS IS KEY FOR YOUR BULLET POINT ISSUE
          ul: ({...props}) => (
            <ul style={{ 
              marginLeft: '20px',  // Indent the list
              marginBottom: '12px',
              textAlign: 'left',   // Ensure left align
              listStylePosition: 'outside'  // Bullets outside
            }} {...props} />
          ),
          ol: ({...props}) => (
            <ol style={{ 
              marginLeft: '20px', 
              marginBottom: '12px',
              textAlign: 'left'
            }} {...props} />
          ),
          li: ({...props}) => (
            <li style={{ marginBottom: '4px', textAlign: 'left' }} {...props} />
          ),
          
          // Bold and italic
          strong: ({...props}) => (
            <strong style={{ fontWeight: 'bold' }} {...props} />
          ),
          em: ({...props}) => (
            <em style={{ fontStyle: 'italic' }} {...props} />
          )
        }}
      >
        {llmResponse}
      </ReactMarkdown>
    </div>
  );
}

function App() {
  const [input, setInput] = useState('')
  const [result, setResult] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async () => {
    if (!input.trim()) return

    setLoading(true)
    try {
      const { data } = await apiClient.post('/api/process', { input })
      setResult(data.result)
    } catch (error) {
      console.error('Error:', error)
      setResult('Error processing request')
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <h1>NaviHealth</h1>
      <div className="card">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter your health query"
          onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
        />
        <button onClick={handleSubmit} disabled={loading}>
          {loading ? 'Processing...' : 'Search'}
        </button>
        {result && (
          <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#1a1a1a', borderRadius: '4px' }}>
            {loading ? '' : <ChatResponse llmResponse={result} />}
          </div>
        )}
      </div>
    </>
  )
}

export default App
