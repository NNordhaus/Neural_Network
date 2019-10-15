using NeuralNetwork.Classes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Tools
{
    public static class Extensions
    {
        public static double MinX = double.MaxValue;
        public static double MaxX = double.MinValue;
        public static double Sigmoid(this double x, int inputCount)
        {
            //MinX = Math.Min(x, MinX);
            //MaxX = Math.Max(x, MaxX);

            double max = (double)inputCount;
            double min = max * -1.0;

            return x < min ? 0.0 : x > max ? 1.0 : 1.0 / (1.0 + Math.Exp(-x));
        }

        public static double SoftArgMax(this IList<double> values)
        {
            int len = 0;
            double max = double.NegativeInfinity;
            foreach (double value in values)
            {
                ++len;
                if (value > max)
                    max = value;
            }

            if (len == 0)
                return -1.0;
            else if (double.IsNegativeInfinity(max))
            {
                return values[Network.Random.Next(len)];
            }

            double total = values.Sum(value => Math.Exp(value - max));

            // Loop just in case due to roundoff we don't choose anything in first pass -- very unlikely.
            for (; ; )
            {
                double r;
                r = Network.Random.NextDouble() * total;

                int i = 0;
                foreach (double value in values)
                {
                    r -= Math.Exp(value - max);
                    if (r <= 0)
                        return values[i];
                    ++i;
                }
            }
        }

        public static double ReLU(this double x, double inputCount)
        {
            //MinX = Math.Min(x, MinX);
            //MaxX = Math.Max(x, MaxX);
            //double max = inputCount;
            if(x < 0.0d)
            {
                return 0.5d * x;
            }
            return Math.Min(inputCount, x);
            //return Math.Max(0.0d,Math.Min(max,x));
        }

        public static double Derivative(this double x)
        {
            return x * (1 - x);
        }

        public static Random rng = new Random(DateTime.Now.Millisecond * 13);
        public static void Shuffle<T>(this IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        public static double Median(this IList<double> list)
        {
            int n = list.Count;
            list = list.OrderBy(a => a).ToList();
            int middle = n / 2;
            return list[middle];
        }

        public static bool DoesBestCandidateMatch(this IList<NeuralNetwork.Classes.Neuron> outputLayer, IList<double> targets)
        {
            int targetIndex = targets.IndexOf(targets.Max());
            var outputs = outputLayer.Select(n => n.Value).ToList();
            return outputs.IndexOf(outputs.Max()) == targetIndex;
        }

        /// <summary>
        /// Compares the contents of two arrays, down to 0.0001 accuracy
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static bool ContentsEqual(this double[] a, double[] b)
        {
            if(a.Length != b.Length)
            {
                return false;
            }

            for(int i = 0; i < a.Length; i++)
            {
                if(Math.Abs(a[i] - b[i]) > 0.0001)
                {
                    return false;
                }
            }

            return true;
        }
    }
}