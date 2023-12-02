import { Line } from 'react-chartjs-2';
import {Chart} from 'chart.js/auto';

export function prepareChargeChartData(data){
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

export function prepareChargeChart(data){
    const charge_chart_data = prepareChargeChartData(data)
    // const chart = new Chart({data: charge_chart_data})
    // Line data={charge_chart_data}
    const options = {
        scales: {
          x: {
            position: 'bottom',
          },
        },
        plugins: {
          legend: {
            display: true, // Set to true to show the legend, false to hide it
          },
        },
        // Add other options as needed
      };
    return <Line data={charge_chart_data} options ={options} />

}