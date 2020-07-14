using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Text;

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TimeSeries;
using Microsoft.ML.Transforms.TimeSeries;

namespace btc
{

    class BTC_TimeSeries
    {
        [LoadColumn(0)]
        public int TimeStamp {get;set;}

        [LoadColumn(1)]
        public float Price {get;set;}
        [LoadColumn(2)]
        public float Amount {get;set;}
    }

    class PredictedSeries
    {
        public float[] ForecastedPrice { get; set; }
        public float[] ConfidenceLowerBound { get; set; }
        public float[] ConfidenceUpperBound { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();
            
            
            int c= 0;
            float predictedPrice = 0;

            while(c < 10)
            {

                IEnumerable<BTC_TimeSeries> data = GetTrainingData(mlContext);

                
                    IDataView trainingData = mlContext.Data.LoadFromEnumerable<BTC_TimeSeries>(data);
                    var offset = TimeSpan.FromSeconds(data.Last().TimeStamp);
                    var epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
                    DateTime lastTime = epoch.Add(offset).ToLocalTime();

                    int seriesLength = data.Count();

                    var estimator = mlContext.Forecasting.ForecastBySsa(outputColumnName: nameof(PredictedSeries.ForecastedPrice),
                                    inputColumnName: nameof(BTC_TimeSeries.Price),
                                    windowSize: 360,
                                    seriesLength: seriesLength,
                                    trainSize: seriesLength,
                                    horizon: 5,
                                    confidenceLevel: 0.95f,
                                    confidenceLowerBoundColumn: nameof(PredictedSeries.ConfidenceLowerBound),
                                    confidenceUpperBoundColumn: nameof(PredictedSeries.ConfidenceUpperBound)
                    );
             
                Console.WriteLine("Training model");
                ITransformer forecastTransformer = estimator.Fit(trainingData);
                
                TimeSeriesPredictionEngine<BTC_TimeSeries, PredictedSeries> forecastEngine = forecastTransformer.CreateTimeSeriesEngine<BTC_TimeSeries, PredictedSeries>(mlContext);

                PredictedSeries predictions = forecastEngine.Predict();
                Console.WriteLine("last read time {0}: last read price {1}", lastTime, data.Last().Price);
                for(int i = 0; i < predictions.ForecastedPrice.Count(); i++)
                {
                    lastTime = lastTime.AddMinutes(15);
                    Console.WriteLine("{0} price: {1}, low: {2}, high: {3}", lastTime, predictions.ForecastedPrice[i].ToString(), predictions.ConfidenceLowerBound[i].ToString(), predictions.ConfidenceUpperBound[i].ToString());
                }

                float delta = predictedPrice - data.Last().Price;
                Console.WriteLine("Delta, {0}", delta);
                predictedPrice = predictions.ForecastedPrice[0];
                forecastEngine.CheckPoint(mlContext, "forecastmodel.zip");

                System.Threading.Thread.Sleep(300000);
                c++;
            }
            

            Console.Read();
        }

        static IEnumerable<BTC_TimeSeries> GetTrainingData(MLContext mlContext)
        {
            Console.WriteLine("Downloading history");

            using(HttpClient client = new HttpClient())
            {
               var response = client.GetAsync("http://api.bitcoincharts.com/v1/trades.csv?symbol=bitstampUSD").Result;

                var results = response.Content.ReadAsStringAsync().Result;


                using (FileStream fileStream = File.Create("btc.csv"))
                {
                    byte[] bytes = Encoding.ASCII.GetBytes(results);
                    fileStream.Write(bytes, 0, bytes.Length);
                }
                
            }
            
            
            IDataView trainingData = mlContext.Data.LoadFromTextFile<BTC_TimeSeries>("btc.csv", hasHeader: false, separatorChar: ',');
            IEnumerable<BTC_TimeSeries> data = mlContext.Data.CreateEnumerable<BTC_TimeSeries>(trainingData, false, true);
            data = data.ToList().OrderBy( o => o.TimeStamp);
            
            return data;
        }

    }
}
