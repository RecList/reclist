
// Returns a random number between min (inclusive) and max (exclusive)
function getRandomArbitrary(min, max) {
    return Math.random() * (max - min) + min;
}

var ctx_bars = document.getElementById('brand_cosine_distance_canvas').getContext('2d');
var cos_distance_chart = REC_CHARTS['brand_cosine_distance']
var cos_datasets = [];
for(var i = 0; i < cos_distance_chart.datasets.length; i ++) {
    var _d = cos_distance_chart.datasets[i]
    var r = getRandomArbitrary(0, 255)
    var g = getRandomArbitrary(0, 255)
    var b = getRandomArbitrary(0, 255)
    _d ['fill'] = true
    _d['backgroundColor'] = 'rgba(' + r + ',' + g + ',' + b + ',' + '0.3)'
    _d['borderColor'] = 'rgb(' + r + ',' + g + ',' + b + ',' + '0.3)'
    _d['borderWidth'] = 1
    cos_datasets.push(_d);
}
var formatted_labels = [];
for(var i = 0; i < cos_distance_chart.labels.length; i++) {
    formatted_labels.push(cos_distance_chart.labels[i].toFixed(3));
}
var chart_bars = new Chart(ctx_bars,
    {
        "type":"bar",
        "data":{
            "labels":  formatted_labels,
            "datasets": cos_datasets
        },
        "options": {
            "scales":
                {"yAxes":
                        [
                            {
                                "ticks": {"beginAtZero": true},
                                scaleLabel: { labelString: 'cos. dist.'}
                            }
                            ]
                    }
            }
    });