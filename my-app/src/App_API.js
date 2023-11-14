
import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import Chart from 'chart.js/auto';

// export default App_API;

function App_API() {
    const [data, setData] = useState(null);
  
    useEffect(() => {
      fetch("https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items")
        .then(response => response.json())
        .then(json => {
          // console.log(json); // Check the fetched data
          setData(json);
        })
        .catch(error => console.error(error));
    }, []);
  
    const preparePriceChartData = () => {
      const sortedData = data.slice().sort((a, b) => a.id - b.id);
      
      const last_Data_value = sortedData[sortedData.length -1]
      
      const past_times = sortedData.map(item => item.time);
      let future_times = Object.keys(last_Data_value.fc_spread)
      future_times = future_times.sort((a, b) => a.localeCompare(b));

      const labels =[...past_times, ...future_times]
      console.log(last_Data_value.fc_spread)
      const actual_price = sortedData.map(item => item.imba_price);
      const forecast_high = [];
      const forecast_low = [];

      future_times.forEach((time) => {
        forecast_high.push(last_Data_value.fc_spread[time][0]);
        forecast_low.push(last_Data_value.fc_spread[time][1]);
      });
      // const forecast_low = last_Data_value.map(item => item.fc_spread[1]);
  
    
      return {
        labels,
        datasets: [
          {
            label: 'IMBA Price',
            data: actual_price,
            fill: false,
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1
          },
          {
            label : "forecast high",
            data: [...Array(past_times.length-1).fill(null), ...forecast_high],
            fill: false,
            borderColor: 'rgba(192, 75, 192,0)',
            tension: 0.1
          },
          {
            label: "forecast low",
            data: [...Array(past_times.length -1).fill(null), ...forecast_low],
            fill: "-1",
            borderColor: 'rgba(0, 75, 192,0)',
            tension: 0.1
          }
        ]
      };
    };


    
    const options = {
      scales: {
        y: {
          beginAtZero: true,
        },
      },
      plugins: {
        legend: {
          labels: {
            // Display only required labels in the legend
            filter: (legendItem, chartData) => {
              return ![
                'forecast high',
                'forecast low'
              ].includes(legendItem.text);
            },
          },
        },
      },
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

    const values = [
      { name: 'Charge cost', day: 100, month: 300,year: 1000 },
      { name: 'Discharge revenue', day: 150, month: 200,year: 800 },
      { name: 'Profit', day: 50, month: 100,year: 200 },
      // Add other specific numbers related to your data
    ];
  
    const renderProfitTable = () => {
      return (
        <table style={{ border: '1px solid black', padding: '10px' }}>
          <thead>
            <tr>
              <th>Imbalance price exposure (euro)</th>
              <th>Day</th>
              <th>Month</th>
              <th>Year</th>
            </tr>
          </thead>
          <tbody>
            {values.map((item, index) => (
              <tr key={index}>
                <td>{item.name}</td>
                <td>{item.day}</td>
                <td>{item.month}</td>
                <td>{item.year}</td>
              </tr>
            ))}
          </tbody>
        </table>
      );
    };
  
    return (
      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'flex-start' }}>
        <div style={{ marginRight: '20px' }}>
          {data ? (
            <div style={{ width: '600px', height: '300px', margin: '10px' }}>
              <Line data={preparePriceChartData()} />
            </div>
          ) : (
            'Loading...'
          )}
  
          {data ? (
            <div style={{ width: '600px', height: '300px', margin: '10px' }}>
              <Line data={prepareChargeChartData()} options={options} />
            </div>
          ) : (
            'Loading...'
          )}
        </div>
  
        <div style={{ width: '600px', height: '300px', margin: '10px' }}>
          {renderProfitTable()}
        </div>
      </div>
    );
  }
    

  
    

    
    export default App_API;