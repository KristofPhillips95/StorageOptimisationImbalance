import { Line } from 'react-chartjs-2';
import {Chart} from 'chart.js/auto';

export function prepareChargeChartData(data){
    if (!data) return { labels: [], datasets: [] };
    
    const sortedData = data.slice().sort((a, b) => a.id - b.id);
    const last_Data_value = sortedData[sortedData.length - 1];

    let future_times = Object.keys(last_Data_value.si_quantile_fc);
    future_times = future_times.sort((a, b) => a.localeCompare(b));
    const past_times_known = sortedData.map(item => item.last_si_time);
    const past_times_unknown = last_Data_value.unkown_times

    const charge_fc = last_Data_value.charge;
    const soc_fc = last_Data_value.soc;
    const discharge_fc = last_Data_value.discharge;

    const nb_ts_before_fc = past_times_known.length+past_times_unknown.length
    const knownElements = sortedData.slice(-nb_ts_before_fc);
    const charge_known = knownElements.map(element => element.charge[0]);
    const discharge_known = knownElements.map(element => element.discharge[0]);
    const soc_known = knownElements.map(element => element.soc[0]);



    const labels = [...past_times_known,...past_times_unknown,...future_times]



    return {
      labels,
      datasets: [
        {
          label: 'Charge',
          data: [...charge_known,...charge_fc],
          fill: false,
          borderColor: 'rgb(192, 75, 192)',
          tension: 0.1
        },
        {
          label: 'Discharge',
          data: [...discharge_known,...discharge_fc],
          fill: false,
          borderColor: 'rgb(192, 75, 0)',
          tension: 0.1
        },
        {
          label: 'State of charge',
          data: [...soc_fc,...soc_fc],
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