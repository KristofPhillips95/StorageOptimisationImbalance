import logo from './logo.svg';
import './App.css';
import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

function MyButton() {
  return (
    <button>
      I'm a button
    </button>
  );
}

function MyApp() {
  return (
    <div>
      <h1>Welcome to my app</h1>
      <MyButton />
    </div>
  );
}




function App_API() {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch("https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items")
      .then(response => response.json())
      .then(json => setData(json))
      .catch(error => console.error(error));
  }, []);

  const prepareChartData = () => {
    if (!data) return { labels: [], datasets: [] };
    
    const labels = data.map(item => item.time); // Assuming 'time' field is available in your data
    const values = data.map(item => item.imba_price); // Assuming 'imba_price' field for the y-axis
    
    return {
      labels,
      datasets: [
        {
          label: 'IMBA Price',
          data: values,
          fill: false,
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1
        }
      ]
    };
  };

  return (
    <div>
      {data ? (
        <Line data={prepareChartData()} />
      ) : (
        'Loading...'
      )}
    </div>
  );
}

const YourComponent = () => {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch('https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items/1') // Replace this URL with your API endpoint
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        setData(data);
      })
      .catch(error => {
        console.error('Error fetching data:', error);
      });
  }, []);

  return (
    <div>
      <h1>Visualizing Data</h1>
      <ul>
        {data.map(item => (
          <li key={item.id}>{item.charge}</li>
        ))}
      </ul>
    </div>
  );
};

export default App_API;





