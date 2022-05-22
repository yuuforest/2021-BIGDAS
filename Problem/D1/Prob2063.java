package Problem.D1;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.StringTokenizer;

// 중간값 찾기
public class Prob2063 {
    public static void main(String args[]) throws Exception
	{
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringBuilder sb = new StringBuilder();

		int T = Integer.parseInt(br.readLine());

        // OPTION 1 : 100ms 18504KB

		StringTokenizer st = new StringTokenizer(br.readLine(), " ");
		int[] intArray = new int[T];

		for (int i = 0; i < T; i++) {
			intArray[i] = Integer.parseInt(st.nextToken());
		}

		Arrays.sort(intArray);
		sb.append(intArray[intArray.length/2]);

		System.out.println(sb);
		br.close();
	}
}
