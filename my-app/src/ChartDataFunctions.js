import { Line } from 'react-chartjs-2';

export function prepareChart(data,type) {
    console.log("Preparing chart")
    const chart_data = prepareChartData(data,type);
  
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
                return legendItem.datasetIndex === 0 ||legendItem.datasetIndex === 4; // Show only the first legend label
              },
            },
          },
        },
        // Add other options as needed
      };
  
    return <Line data={chart_data} options={options} />;
  };
function prepareChartData(data,type) {
    if (data === null || data === undefined) {
        console.log("Data is still null or undefined. Fetching in progress or failed.");
        return { labels: [], datasets: [] };
        }
    const sortedData = data.slice().sort((a, b) => a.id - b.id).slice(-12);
    const past_times_known = sortedData.map(item => item.last_si_time);
    const last_Data_value = sortedData[sortedData.length - 1];
    const past_times_unknown = last_Data_value.unkown_times
    const nb_ts_before_fc = past_times_known.length+past_times_unknown.length
    // console.log(past_times_known)
    // console.log(past_times_unknown)
    // console.log(nb_ts_before_fc)
    let future_times = Object.keys(last_Data_value.si_quantile_fc);
    future_times = future_times.sort((a, b) => a.localeCompare(b));
    // console.log(future_times)
    const quantiles = last_Data_value.quantiles

    const x_labels = [...past_times_known,...past_times_unknown, ...future_times];
    // const x_labels = [...future_times];
    let known_values;
    if (type == "Imba") {
        known_values = sortedData.map(item => item.last_si_value);
        }
    else{
        known_values = sortedData.map(item => item.last_imbPrice_value);
    }
    // console.log(known_values)

    // console.log(x_labels)
    // console.log(past_times_known)
    const chartDataSets = [
        {
        label: 'Historical',
        data: known_values,
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
        },
        ...CreateForecastDataSet(future_times,quantiles,last_Data_value,nb_ts_before_fc,type),

    ]
    if(type == "Price"){
        chartDataSets.push(
            {
                label: 'Avg fc',
                data: [...Array(nb_ts_before_fc).fill(null),...last_Data_value.avg_price_fc],
                fill: false,
                borderColor: 'rgb(75, 192, 20)',
                tension: 0.1,
                borderWidth: 1,
                }
        )
    }
    return {
        labels:x_labels,
        datasets: chartDataSets,
    };
};
function getDatasetColors(future_times,quantiles){
const datasetColors = future_times.map((time, index) => {
    let q_prev; 
    if (index > 0) {
    q_prev = quantiles[index-1]
    }
    else{
    q_prev = 0
    }
    const alpha = -(q_prev - quantiles[index])*3; // Adjust as needed

    return `rgba(105, 105, 105, ${alpha})`; // Dark grey with varying transparency
});
return datasetColors
};
function CreateForecastDataSet(future_times,quantiles,last_Data_value,nb_ts_before_fc,type){
    const forecasts = [];
    const nb_qs = quantiles.length
    // // Generate an array of colors with different transparencies
    // const datasetColors = future_times.map((time, index) => {
    //   const alpha = -((Math.abs(index/nb_qs -1/2) -1 )) * (-((Math.abs(index/nb_qs -1/2) -1 )))  ; // Adjust as needed
    //   return `rgba(105, 105, 105, ${alpha})`; // Dark grey with varying transparency
    // });
      // Generate an array of colors with different transparencies
  
    const datasetColors = getDatasetColors(future_times,quantiles)
    // Loop over quantiles
    for (let i = 0; i < nb_qs ; i++) {
      // Use an index to create a unique legend label for each quantile
        let thisq_data = [];        
        future_times.forEach((time) => {
            let fcSpreadValue;
            if(type == "Imba"){
                 fcSpreadValue = last_Data_value.si_quantile_fc[time][i];
            }
            else{
                fcSpreadValue = last_Data_value.quantile_price_fc[time][i];
            }
            // console.log(time,thisq_data)
            thisq_data.push(fcSpreadValue);
        });
        // console.log(thisq_data)
    
        // Choose color based on the index (you can modify this logic as needed)
        const colorIndex = i % datasetColors.length;
        const backgroundColor = datasetColors[colorIndex];
    
        const dataset = {
            data: [...Array(nb_ts_before_fc).fill(null),...thisq_data],
            // data: [...thisq_data],
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
    return forecasts
};