package Problem.D1;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

// 평균값 구하기
public class Prob2071 {
    public static void main(String args[]) throws Exception
	{
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringBuilder sb = new StringBuilder();

		int T = Integer.parseInt(br.readLine());

        // OPTION 1 : 111ms 19872KB
        
		for (int i = 0; i < T; i++) {

			StringTokenizer st = new StringTokenizer(br.readLine(), " ");
			int sum = 0;

			for (int j = 0; j < 10; j++) {
				sum += Integer.parseInt(st.nextToken()); 
			}

			sb.append("#" + (i+1) + " " + String.format("%.0f", sum/10.0) + "\n");
		}

		System.out.println(sb);
		br.close();
	}
}
