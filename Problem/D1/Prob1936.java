package Problem.D1;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

// 1대1 가위바위보
public class Prob1936 {
    public static void main(String args[]) throws Exception
	{
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringBuilder sb = new StringBuilder();

        // OPTION 1 : 97ms 18584KB
        
		StringTokenizer st = new StringTokenizer(br.readLine(), " ");

		int a = Integer.parseInt(st.nextToken());
		int b = Integer.parseInt(st.nextToken());

		int value = Math.abs(a - b);

		if(value == 1) {
			if ((a == 3) || (a == 2 && b == 1)) sb.append("A");
			else if ((a == 1) || (a == 2 && b == 3)) sb.append("B");
		} else {
			if (a == 3) sb.append("B");
			else sb.append("A");
		}

		System.out.println(sb);
		br.close();
	}
}
