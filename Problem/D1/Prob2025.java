package Problem.D1;

import java.io.BufferedReader;
import java.io.InputStreamReader;

// N줄 덧셈
public class Prob2025 {
    public static void main(String args[]) throws Exception
	{
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringBuilder sb = new StringBuilder();

        // OPTION 1 : 105ms 18804KB

		int T = Integer.parseInt(br.readLine());
		int sum = 0;

		for (int i = 1; i <= T; i++) {
			sum += i;
		}

		sb.append(sum);
		System.out.println(sb);
	}
}
