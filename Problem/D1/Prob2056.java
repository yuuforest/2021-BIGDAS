package Problem.D1;

import java.io.BufferedReader;
import java.io.InputStreamReader;

// 연월일 달력
public class Prob2056 {
    public static void main(String args[]) throws Exception
	{
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringBuilder sb = new StringBuilder();

		int T = Integer.parseInt(br.readLine());
        
        // OPTION 1 : 101ms 18340KB

		for (int i = 0; i < T; i++) {

			String str = br.readLine();
			boolean check = true;

			int month = Integer.parseInt(str.substring(4, 6));
			int day = Integer.parseInt(str.substring(6, 8));

			if(month < 1 || month > 12) check = false;
			else {
				switch (month){
					case 1: case 3: case 5: case 7: case 8: case 10: case 12:
						if (day < 1 || day > 31) check = false; break;
					case 2:
						if (day < 1 || day > 28) check = false; break;
					case 4: case 6: case 9: case 11:
						if (day < 1 || day > 30) check = false; break;
					default:
						break;
				}
			}
			
			if (check) sb.append("#" + (i+1) + " " + str.substring(0, 4) + "/" + str.substring(4, 6) + "/" + str.substring(6, 8) + "\n");
			else sb.append("#" + (i+1) + " -1\n"); 
		}

		System.out.println(sb);
		br.close();
	}
}
