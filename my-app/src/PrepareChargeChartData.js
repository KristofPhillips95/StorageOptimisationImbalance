import { Line,Bar } from 'react-chartjs-2';
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
    console.log(knownElements)
    return {
      labels,
      datasets: [
        {
          label: 'SOC (Known)',
          data: [...soc_known, ...Array(nb_ts_before_fc - soc_known.length).fill(null)],
          fill: false,
          borderColor: 'rgb(0, 75, 192)',
          tension: 0.1,
          yAxisID: 'A', // Assign to the soc y-axis
        },
        {
          label: 'SOC (Forecast)',
          data: [...Array(nb_ts_before_fc - 1).fill(null), ...soc_fc],
          fill: false,
          borderColor: 'rgb(0, 75, 192)',
          tension: 0.1,
          borderDash: [5, 5],
          yAxisID: 'A', // Assign to the soc y-axis
        },
        {
          label: 'Charge',
          data: [...charge_known, ...charge_fc],
          type: 'bar',
          backgroundColor: 'rgb(192, 75, 192)',
          yAxisID: 'B', // Assign to the cd y-axis
          barThickness: 20, // Set to 'flex' to occupy the full available space
        },
        {
          label: 'Discharge',
          data: [...discharge_known.map(value => -value), ...discharge_fc.map(value => -value)],
          type: 'bar',
          backgroundColor: 'rgb(192, 75, 0)',
          yAxisID: 'B', // Assign to the cd y-axis
          barThickness: 20, // Set to 'flex' to occupy the full available space
        },
      ]
    };
  };
function prepareChargeChartOptions() {
  console.log("Preparing charge chart options");

  return {
    plugins: {
      title: {
        display: true,
        text: 'Storage operation',
        font: {
          size: 18, // Adjust the font size as needed
        },
      },
    },
      scales: {
          A: {
              type: 'linear',
              position: 'left',
              min:0,
              max:4,
              title: {
                display: true,
                text: "SOC (MWh)",
                font: {
                    size: 14, // Adjust the font size as needed
                },
            },
          },
          B: {
              type: 'linear',
              position: 'right',
              min:-2,
              max:2,
              title: {
                display: true,
                text: "(Dis)charge (MW)",
                font: {
                    size: 14, // Adjust the font size as needed
                },
            },
          }
      }
  }
}
  
export function prepareChargeChart(data){
    const charge_chart_data = prepareChargeChartData(data)
    let charge_chart_options;
    charge_chart_options = prepareChargeChartOptions()
    // const chart = new Chart({data: charge_chart_data})
    // Line data={charge_chart_data}
    return <Line data={charge_chart_data} options = {charge_chart_options} />

}