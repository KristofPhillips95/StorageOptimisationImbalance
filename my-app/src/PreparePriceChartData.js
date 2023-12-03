import { Line } from 'react-chartjs-2';

export function prepareImbalanceChartData(data) {
    if (data === null || data === undefined) {
        console.log("Data is still null or undefined. Fetching in progress or failed.");
        return { labels: [], datasets: [] };
      }
    const sortedData = data.slice().sort((a, b) => a.id - b.id);
  
    const last_Data_value = sortedData[sortedData.length - 1];
  
    const past_times = sortedData.map(item => item.time);
    let future_times = Object.keys(last_Data_value.fc_spread);
    future_times = future_times.sort((a, b) => a.localeCompare(b));
  
    const x_labels = [...past_times, ...future_times];
    const actual_imba = sortedData.map(item => item.imba_price);
  
    const forecasts = [];
    const nb_qs =9
    // Generate an array of colors with different transparencies
    const datasetColors = future_times.map((time, index) => {
      const alpha = -((Math.abs(index/nb_qs -1/2) -1 )) * (-((Math.abs(index/nb_qs -1/2) -1 )))  ; // Adjust as needed
      return `rgba(105, 105, 105, ${alpha})`; // Dark grey with varying transparency
    });
          
    // Loop over quantiles
    for (let i = 0; i < nb_qs ; i++) {
      // Use an index to create a unique legend label for each quantile
      const label = `forecast q ${i + 1}`;
      let thisq_data = [];
    
      future_times.forEach((time) => {
        const fcSpreadValue = last_Data_value.fc_spread[time][i];
        thisq_data.push(fcSpreadValue);
      });
    
      // Choose color based on the index (you can modify this logic as needed)
      const colorIndex = i % datasetColors.length;
      const backgroundColor = datasetColors[colorIndex];
    
      const dataset = {
        data: [...Array(past_times.length-1).fill(null),actual_imba[actual_imba.length - 1],...thisq_data],
        fill: "-1",
        backgroundColor,
        borderColor: backgroundColor, // Set borderColor to make lines invisible
        pointRadius: 0, // Set pointRadius to 0 to remove data point markers
        borderWidth: 0, // Adjust the line thickness
        tension: 0.1,
        label: "Forecast",
      };
    
      forecasts.push(dataset);
    }
    // console.log(x_labels)
    console.log(actual_imba)
    const imbaChartDataSets = [
      {
        label: 'Historical',
        data: actual_imba,
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      },
      ...forecasts
    ]
    return {
      labels:x_labels,
      datasets: imbaChartDataSets,
    };
  };
  export function preparePriceChart(data) {
    const price_chart_data = prepareImbalanceChartData(data);
  
    const options = {
        scales: {
          x: {
            position: 'bottom',
          },
        },
        plugins: {
          legend: {
            display: true, // Display the legend
            position: 'top', // Set the position of the legend
            labels: {
              filter: function (legendItem, chartData) {
                return legendItem.datasetIndex === 0 ||legendItem.datasetIndex === 1 ; // Show only the first legend label
              },
            },
          },
        },
        // Add other options as needed
      };
  
    return <Line data={price_chart_data} options={options} />;
  }