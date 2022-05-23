package Problem.D1;

import java.io.BufferedReader;
import java.io.InputStreamReader;

// 간단한 N의 약수
public class Prob1933 {
    public static void main(String args[]) throws Exception
	{
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringBuilder sb = new StringBuilder();

        // OPTION 1 : 101ms 18672KB

		int T = Integer.parseInt(br.readLine());

		for (int i = 1; i <= T; i++) {
			if (T % i == 0) sb.append(i + " ");
		}

		System.out.println(sb);
		br.close();
	}
}
