$(document).ready(function(){
    $("#state").keypress(function (e) {
        if (e.which === 13 && !e.shiftKey) {
          e.preventDefault();
          $(this).closest("form").submit();
        }
      });
  });
    