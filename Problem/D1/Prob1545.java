package Problem.D1;

import java.io.BufferedReader;
import java.io.InputStreamReader;

// 거꾸로 출력해 보아요
public class Prob1545 {
    public static void main(String args[]) throws Exception
	{
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringBuilder sb = new StringBuilder();

        // OPTION 1 : 101ms 18408KB
        
		int T = Integer.parseInt(br.readLine());

		for (int i = T; i >= 0; i--) {
			sb.append(i + " ");
		}

		System.out.println(sb);
		br.close();
	}
}
