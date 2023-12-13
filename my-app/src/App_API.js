
import React, { useState, useEffect } from 'react';
import { RenderProfitTable } from './RenderProfitTable';
import { prepareChargeChart } from './PrepareChargeChartData';
import { prepareChart } from './ChartDataFunctions';

function fetchData(link) {
  return fetch(link)
  .then(response=> response.json())
  .catch(error=> {
    console.error(error);
    throw error
   }
  )
}

function useDataEffect(endpoint, setData) {
  useEffect(() => {
    fetchData(endpoint)
      .then(json => setData(json))
      .catch(error => console.error(error));
  }, [endpoint, setData]);
}

function App_API() {
  // console.log("Entered APP_API")
    const [data, setData] = useState(null);
    const [data_lts, setData_lts] = useState(null);
  

    useDataEffect("https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items", setData);
    useDataEffect("https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/lts_items", setData_lts);
    console.log(data_lts)
    
  const imba_chart_2 = prepareChart(data,"Imba")
  const price_chart_2 = prepareChart(data,"Price")
  const charge_chart = prepareChargeChart(data)

    

    const values = [
      { name: 'Charge cost', day: 100, month: 300,year: 1000 },
      { name: 'Discharge revenue', day: 150, month: 200,year: 800 },
      { name: 'Profit', day: 50, month: 100,year: 200 },
      // Add other specific numbers related to your data
    ];
  

    if (data === null) {
      return <div>Loading...</div>;
    }
    return (
      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'flex-start' }}>
        <div style={{ marginRight: '20px' }}>
          <div style={{ width: '600px', height: '300px', margin: '10px' }}>{imba_chart_2}</div>
          <div style={{ width: '600px', height: '300px', margin: '10px' }}>{price_chart_2}</div>
          <div style={{ width: '600px', height: '300px', margin: '10px' }}>{charge_chart}</div>
        </div>
        <div style={{ width: '600px', height: '300px', margin: '10px' }}>
          {RenderProfitTable(values,data_lts)}
        </div>
      </div>
    );
  }

export default App_API;