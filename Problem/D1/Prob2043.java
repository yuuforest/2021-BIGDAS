package Problem.D1;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

public class Prob2043 {
    public static void main(String args[]) throws Exception
	{
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringBuilder sb = new StringBuilder();

        // OPTION 1 : 107ms 18452KB

		StringTokenizer st = new StringTokenizer(br.readLine(), " ");

		int P = Integer.parseInt(st.nextToken());
		int K = Integer.parseInt(st.nextToken());

		sb.append(P-K+1);

		System.out.println(sb);
		br.close();
	}
}
