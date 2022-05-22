package Problem.D1;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

// 큰 놈, 작은 놈, 같은 놈
public class Prob2070 {
    public static void main(String args[]) throws Exception
	{
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringBuilder sb = new StringBuilder();

		// OPTION 1 : 95ms 18380KB

		int T = Integer.parseInt(br.readLine());

		for (int i = 0; i < T; i++) {

			StringTokenizer st = new StringTokenizer(br.readLine(), " ");
			
			int a = Integer.parseInt(st.nextToken());
			int b = Integer.parseInt(st.nextToken());

			if (a > b) sb.append("#" + (i+1) + " >\n");
			else if (a < b) sb.append("#" + (i+1) + " <\n");
			else sb.append("#" + (i+1) + " =\n");
		}

		System.out.println(sb);
		br.close();
	}
}
