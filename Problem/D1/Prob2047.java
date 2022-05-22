package Problem.D1;

import java.io.BufferedReader;
import java.io.InputStreamReader;

// 신문 헤드라인
public class Prob2047 {
    public static void main(String args[]) throws Exception
	{
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringBuilder sb = new StringBuilder();

        // OPTION 1 : 103ms 18388KB
        
		sb.append(br.readLine().toUpperCase());

		System.out.println(sb);
		br.close();
	}
}
