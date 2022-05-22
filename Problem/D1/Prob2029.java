package Problem.D1;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

// 몫과 나머지 출력하기
public class Prob2029 {
    public static void main(String args[]) throws Exception
	{
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringBuilder sb = new StringBuilder();

        // OPTION 1 : 97ms 18708KB

		int T = Integer.parseInt(br.readLine());

		for (int i = 0; i < T; i++) {

			StringTokenizer st = new StringTokenizer(br.readLine(), " ");

			int a = Integer.parseInt(st.nextToken());
			int b = Integer.parseInt(st.nextToken());

			sb.append("#" + (i+1) + " " + (a/b) + " " + (a%b) + "\n");
		}

		System.out.println(sb);
		br.close();
	}
}
