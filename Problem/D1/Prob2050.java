package Problem.D1;

import java.io.BufferedReader;
import java.io.InputStreamReader;

// 알파벳을 숫자로 변환
public class Prob2050 {
    public static void main(String args[]) throws Exception
	{
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringBuilder sb = new StringBuilder();

        // OPTION 1 : 106ms 18228KB
        
		String str = br.readLine();

		for (int i = 0; i < str.length(); i++) {
			sb.append((str.charAt(i) - 'A' + 1) + " ");
		}

		System.out.println(sb);
		br.close();
	}
}
