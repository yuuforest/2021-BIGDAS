package Problem.D1;

import java.io.BufferedReader;
import java.io.InputStreamReader;

// 더블더블
public class Prob2019 {
    public static void main(String args[]) throws Exception
	{
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringBuilder sb = new StringBuilder();

        // OPTION 1 : 97ms 18696KB
        
		int T = Integer.parseInt(br.readLine());
		int count = 1;

		for (int i = 0; i <= T; i++) {
			sb.append(count + " ");
			count *= 2;
		}

		System.out.println(sb);
		br.close();
	}
}
