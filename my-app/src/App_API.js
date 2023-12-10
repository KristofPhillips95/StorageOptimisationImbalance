
import React, { useState, useEffect } from 'react';
import { renderProfitTable } from './RenderProfitTable';
import { prepareChargeChart } from './PrepareChargeChartData';
import { prepareChart } from './ChartDataFunctions';

function App_API() {
  console.log("Entered APP_API")
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
      // Check if data is not yet available
  if (data === null) {
    return <div>Loading...</div>;
  }


  const imba_chart_2 = prepareChart(data,"Imba")
  const price_chart_2 = prepareChart(data,"Price")
  const charge_chart = prepareChargeChart(data)

    

    const values = [
      { name: 'Charge cost', day: 100, month: 300,year: 1000 },
      { name: 'Discharge revenue', day: 150, month: 200,year: 800 },
      { name: 'Profit', day: 50, month: 100,year: 200 },
      // Add other specific numbers related to your data
    ];
  

  
    return (
      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'flex-start' }}>
        <div style={{ marginRight: '20px' }}>
          {data ? (
            <div style={{ width: '600px', height: '300px', margin: '10px' }}>
              {imba_chart_2}
            </div>
          ) : (
            'Loading...'
          )}
          {data ? (
            <div style={{ width: '600px', height: '300px', margin: '10px' }}>
              {price_chart_2}
            </div>
          ) : (
            'Loading...'
          )}
  
          {data ? (
            <div style={{ width: '600px', height: '300px', margin: '10px' }}>
              {charge_chart}
            </div>
          ) : (
            'Loading...'
          )}
        </div>
  
        <div style={{ width: '600px', height: '300px', margin: '10px' }}>
          {renderProfitTable(values)}
        </div>
      </div>
    );
  }
    


    
    // const options = {
    //   scales: {
    //     y: {
    //       beginAtZero: true,
    //     },
    //   },
    //   plugins: {
    //     legend: {
    //       labels: {
    //         display: false
    //       },
    //     },
    //   },
    // };
  


  
export default App_API;