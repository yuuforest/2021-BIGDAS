package Problem.D1;

import java.io.BufferedReader;
import java.io.InputStreamReader;

// 자릿수 더하기
public class Prob2058 {
    public static void main(String args[]) throws Exception
	{
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringBuilder sb = new StringBuilder();

        // OPTION 1 : 95ms 18656KB
        
		int T = Integer.parseInt(br.readLine());
		int sum = 0;

		while (T > 0) {
			sum += (T%10);
			T /= 10;
		}

		sb.append(sum);

		System.out.println(sb);
		br.close();
	}
}
