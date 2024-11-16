function homeLink() {
  const homeLink = document.getElementById('Home-link');

  homeLink.addEventListener('click', function (event) {
    event.preventDefault();
    window.scroll({ top: 0, behavior: 'smooth' });
  });
}

function aboutLink() {
  const aboutLink = document.getElementById('About-Link');

  aboutLink.addEventListener('click', function (event) {
    event.preventDefault();
    document.getElementById('page3').scrollIntoView({ behavior: 'smooth' });
  });
}

homeLink()
aboutLink()
