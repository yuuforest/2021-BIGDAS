package Problem.D1;

import java.io.BufferedReader;
import java.io.InputStreamReader;

//  스탬프 찍기
public class Prob2046 {
    public static void main(String args[]) throws Exception
	{
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringBuilder sb = new StringBuilder();

		int T = Integer.parseInt(br.readLine());

        // OPTION 1 : 98ms 18340KB
        
		for (int i = 0; i < T; i++) {
			sb.append("#");
		}

		System.out.println(sb);
		br.close();
	}
}
