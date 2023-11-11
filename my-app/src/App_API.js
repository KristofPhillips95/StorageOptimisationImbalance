
import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import Chart from 'chart.js/auto';

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
      
      const sortedData = data.slice().sort((a, b) => a.id - b.id);
    
      const labels = sortedData.map(item => item.time);
      const values = sortedData.map(item => item.imba_price);
      const values_2 = sortedData.map(item => item.imba_prce_fc);
  
      return {
        labels,
        datasets: [
          {
            label: 'IMBA Price',
            data: values,
            fill: false,
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1
          },
          {
            label: 'IMBA Price forecasted',
            data: values_2,
            fill: false,
            borderColor: 'rgb(75, 0, 192)',
            tension: 0.1
          }
        ]
      };
    };
  
    const prepareChargeChartData = () => {
      if (!data) return { labels: [], datasets: [] };
      
      const sortedData = data.slice().sort((a, b) => a.id - b.id);
    
      const labels = sortedData.map(item => item.time);
      const values_3 = sortedData.map(item => item.charge);
      const values_4 = sortedData.map(item => item.soc);

  
      return {
        labels,
        datasets: [
          {
            label: 'Charge',
            data: values_3,
            fill: false,
            borderColor: 'rgb(192, 75, 192)',
            tension: 0.1
          },
          {
            label: 'State of charge',
            data: values_4,
            fill: false,
            borderColor: 'rgb(0, 75, 192)',
            tension: 0.1
          }
        ]
      };
    };
  
    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          {data ? (
            <div style={{ width: '600px', height: '300px', margin: '10px' }}>
              <Line data={prepareChartData()} />
            </div>
          ) : (
            'Loading...'
          )}
    
          {data ? (
            <div style={{ width: '600px', height: '300px', margin: '10px' }}>
              <Line data={prepareChargeChartData()} />
            </div>
          ) : (
            'Loading...'
          )}
        </div>
      );
    }
  

export default App_API